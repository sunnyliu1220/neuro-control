import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as colors
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.decomposition import PCA
import time
import utils

class RNN():
    '''
    Partially adapted from Fatih Dinc et al. Convex optimization of recurrent neural networks for rapid inference of neural dynamics, https://openreview.net/forum?id=GGIA1p9fDT.
    This is a class of RNNs. It has the following attributes:
    - n_rec: number of neurons
    - n_in: number of input dimensions/neurons
    - alpha: discretization time scale
    - sigma_rec: standard deviation of the noise
    - w_in: input weight matrix
    - w_rec: recurrent weight matrix
    - phi: optional nonlinearity
    '''

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
        if ('n_rec' not in kwargs.keys() or self.n_rec is None):
            self.n_rec = 500
        if 'n_in' not in kwargs.keys() or self.n_in is None:
            self.n_in = 0 # Number of input dimensions/neurons
        if 'alpha' not in kwargs.keys() or self.alpha is None:
            self.alpha = 0.01
        if 'sigma_rec' not in kwargs.keys() or self.sigma_rec is None:
            self.sigma_rec = 0.02 / np.sqrt(2) # Standard deviation of the recurrent noise
        if 'w_in' not in kwargs.keys() or self.w_in is None:
            if self.n_in > 0:
                self.w_in = np.random.normal(0, 1, (self.n_rec, self.n_in)) / np.sqrt(self.n_rec)
        if 'w_rec' not in kwargs.keys() or self.w_rec is None:
            self.w_rec = np.random.normal(0, 1, (self.n_rec, self.n_rec)) / np.sqrt(self.n_rec)
        if 'phi' not in kwargs.keys() or self.phi is None:
            self.phi = lambda x:x # Optional nonlinearity

    def forward_step(self,r_in,u = None, noise = None):
        '''
        Runs every snapshot of the input firing rates by one step and returns the updated firing rates over time.
        Inputs:
        - r_in: initial firing rates (..., T, n_rec)
        - u: input throughout time (..., T, n_in)
        - noise: optional given noise of the network (..., T, n_rec)
        '''
        assert r_in.shape[-1] == self.n_rec
        T = r_in.shape[-2]
        n_rec = self.n_rec
        n_in = self.n_in
        alpha = self.alpha
        sigma_rec = self.sigma_rec
        if n_in>0:
            w_in  = self.w_in
        w_rec = self.w_rec
        phi = self.phi
        if noise is None:
            noise = np.random.normal(0,sigma_rec,r_in.shape)

        if u is None:
            if n_in > 0:
                u = np.zeros(r_in.shape[:-1] + self.n_in)
        else: assert r_in.shape[:-1] == u.shape[:-1]

        if u is not None: inflow = np.einsum('...ij,...j->...i', w_in, phi(u)) # Input current
        else: inflow = np.zeros(r_in.shape)
        r_out = (1-alpha) * r_in + alpha * (np.einsum('...ij,...j->...i', w_rec, phi(r_in)) + inflow) + np.sqrt(2*alpha)*noise

        return r_out

    def forecast(self,r_in = None,u = None,T = None, noise = None):
        '''
        Runs the RNN forward in time for T time steps and returns the time evolution of the firing rates.
        Inputs:
        - r_in: initial firing rates
        - u: input throughout time (T, n_in)
        - T: total number of time steps
        - noise: optional given noise of the network (T, n_rec)
        '''
        n_rec = self.n_rec
        n_in = self.n_in
        alpha = self.alpha
        sigma_rec = self.sigma_rec
        if n_in>0:
            w_in  = self.w_in
        w_rec = self.w_rec
        phi = self.phi

        if T is None:
            T = 1000
        if r_in is None:
            r_in = np.random.uniform(-1,1,(1,self.n_rec)) # Initialize firing rates
        if u is None:
            if n_in > 0:
                u = np.zeros((T, self.n_in))
        if noise is None:
            noise = np.random.normal(0,sigma_rec,(T,n_rec))
        
        r = np.zeros((T, n_rec)) # We forecast T-1 times forward, so we end up with T time points
        T_init = r_in.shape[0] # Sometimes the input contains multiple time points, but we only use the last
        r[0:T_init,:] = r_in
        for i in np.arange(T_init-1, T-1):
            r_temp=r[i:i+1,:] # firing rates at time i, keep the shape
            noise_temp = noise[i:i+1,:]
            if u is None: u_temp = None
            else: u_temp = u[i:i+1,:]
                
            # Do the update
            r[i+1,:] = self.forward_step(r_temp,u_temp,noise_temp)

        return r
    
class Graphs():
    '''
    Stores the data of and makes all the graphs in the analysis.
    Attributes:
    - r_gt_all: ground truth firing rates. (T, n)
    - r_gt_obs: observed ground truth firing rates. (T, n_obs)
    - r_pred: predicted firing rates. (T, n)
    - elbos_train: ELBOs of the fitting the SLDS. (n_iter,)
    - elbos_pred: ELBOs of latent state estimation based on new observations. (n_iter,)
    - eigenvals_gt: eigenvalues of the ground truth dynamics matrix. (n,)
    - eigenvals_pred: eigenvalues of the predicted dynamics matrix. (n,)
    TODO: Check if large eigenvalues are cropped out of the graph.
    - rnn: RNN object used to make flow field
    '''
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
        # Make not make sense but intended for future adaptation with sample plots
        if 'r_gt_all_train' not in kwargs.keys() or self.r_gt_all_train is None:
            self.r_gt_all_train = None
        if 'r_gt_all_test' not in kwargs.keys() or self.r_gt_all_test is None:
            self.r_gt_all_test = None
        if 'r_gt_obs_train' not in kwargs.keys() or self.r_gt_obs_train is None:
            self.r_gt_obs_train = None
        if 'r_gt_obs_test' not in kwargs.keys() or self.r_gt_obs_test is None:
            self.r_gt_obs_test = None
        if 'r_pred_train' not in kwargs.keys() or self.r_pred_train is None:
            self.r_pred_train = None
        if 'r_pred_test' not in kwargs.keys() or self.r_pred_test is None:
            self.r_pred_test = None
        if 'elbos_train' not in kwargs.keys() or self.elbos_train is None:
            self.elbos_train = None
        if 'elbos_pred_train' not in kwargs.keys() or self.elbos_pred_train is None:
            self.elbos_pred_train = None
        if 'elbos_pred_test' not in kwargs.keys() or self.elbos_pred_test is None:
            self.elbos_pred_test = None
        if 'eigenvals_gt' not in kwargs.keys() or self.eigenvals_gt is None:
            self.eigenvals_gt = None
        if 'eigenvals_pred' not in kwargs.keys() or self.eigenvals_pred is None:
            self.eigenvals_pred = None
        if 'forecast_gt' not in kwargs.keys() or self.forecast_gt is None:
            self.forecast_gt = None
        if 'forecast_pred' not in kwargs.keys() or self.forecast_pred is None:
            self.forecast_pred = None

    def plot_all(self, path = None):
        '''
        Plot all the graphs we can plot.
        '''
        if path is None:
            path = './graphs/analysis.pdf'
        pp = PdfPages(path)
        props = {'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.5} # Text box style
        time_start = time.perf_counter()
        # First, we plot everything for the training set
        if self.elbos_train is not None:
            fig, ax = self.plot_elbos(self.elbos_train)
            ax.set_title('ELBOs of Training')
            pp.savefig(fig)
            print ('ELBOs of training plotted in {:.2f} seconds'.format(time.perf_counter()-time_start))
            time_start = time.perf_counter()
        if self.elbos_pred_train is not None:
            fig, ax = self.plot_elbos(self.elbos_pred_train)
            fig.text(0., 1., 'Train', ha='left', va='top', bbox=props)
            ax.set_title('ELBOs of Forecasting on the Training Set')
            pp.savefig(fig)
            print ('ELBOs of forecasting on the training set plotted in {:.2f} seconds'.format(time.perf_counter()-time_start))
            time_start = time.perf_counter()
        if self.r_gt_all_train is not None:
            fig, ax = self.plot_firing_rates(self.r_gt_all_train)
            fig.text(0., 1., 'Train', ha='left', va='top', bbox=props)
            ax.set_title('Ground Truth Firing Rates')
            pp.savefig(fig)
            print ('Ground truth firing rates for the training set plotted in {:.2f} seconds'.format(time.perf_counter()-time_start))
            time_start = time.perf_counter()
        if self.r_pred_train is not None:
            fig, ax = self.plot_firing_rates(self.r_pred_train)
            fig.text(0., 1., 'Train', ha='left', va='top', bbox=props)
            ax.set_title('Predicted Firing Rates')
            pp.savefig(fig)
            print ('Prediction firing rates for the training set plotted in {:.2f} seconds'.format(time.perf_counter()-time_start))
            time_start = time.perf_counter()
        if self.r_gt_obs_train is not None:
            fig, ax = self.plot_distribution(self.r_gt_obs_train)
            fig.text(0., 1., 'Train', ha='left', va='top', bbox=props)
            ax.set_title('Observed Ground Truth Distribution')
            pp.savefig(fig)
            print ('Ground truth distribution for the training set plotted in {:.2f} seconds'.format(time.perf_counter()-time_start))
            time_start = time.perf_counter()
            if self.r_pred_train is not None:
                fig, ax_gt, ax_pred = self.plot_firing_rates_comparison(self.r_gt_obs_train, self.r_pred_train)
                fig.text(0., 1., 'Train', ha='left', va='top', bbox=props)
                pp.savefig(fig)
                print ('Firing rates comparison for the training set plotted in {:.2f} seconds'.format(time.perf_counter()-time_start))
                time_start = time.perf_counter()
                fig, ax = self.plot_residuals(self.r_gt_obs_train, self.r_pred_train, True)
                fig.text(0., 1., 'Train', ha='left', va='top', bbox=props)
                pp.savefig(fig)
                print ('Normalized residuals for the training set plotted in {:.2f} seconds'.format(time.perf_counter()-time_start))
                time_start = time.perf_counter()
                fig, ax = self.plot_residuals(self.r_gt_obs_train, self.r_pred_train, False)
                fig.text(0., 1., 'Train', ha='left', va='top', bbox=props)
                pp.savefig(fig)
                print ('Unnormalized residuals for the training set plotted in {:.2f} seconds'.format(time.perf_counter()-time_start))
                time_start = time.perf_counter()
                fig, ax = self.plot_neurons_comparison(self.r_gt_obs_train, self.r_pred_train)
                fig.text(0., 1., 'Train', ha='left', va='top', bbox=props)
                pp.savefig(fig)
                print ('Neurons comparison for the training set plotted in {:.2f} seconds'.format(time.perf_counter()-time_start))
                time_start = time.perf_counter()
        if self.eigenvals_pred is not None or self.eigenvals_gt is not None:
            fig, ax = self.plot_spectrum_comparison(self.eigenvals_gt, self.eigenvals_pred)
            ax.set_title('Spectral Analysis')
            pp.savefig(fig)
            print ('Spectrum comparison plotted in {:.2f} seconds'.format(time.perf_counter()-time_start))
            time_start = time.perf_counter()
        if self.forecast_gt is not None:
            fig, ax = self.plot_dynamics(self.forecast_gt)
            ax.set_title('Dynamics of Ground Truth')
            pp.savefig(fig)
            print ('Dynamics of ground truth plotted in {:.2f} seconds'.format(time.perf_counter()-time_start))
            time_start = time.perf_counter()
        if self.forecast_pred is not None:
            fig, ax = self.plot_dynamics(self.forecast_pred)
            ax.set_title('Dynamics of Prediction')
            pp.savefig(fig)
            print ('Dynamics of prediction plotted in {:.2f} seconds'.format(time.perf_counter()-time_start))
            time_start = time.perf_counter()

        # Then, we plot everything for the test set
        if self.elbos_pred_test is not None:
            fig, ax = self.plot_elbos(self.elbos_pred_test)
            fig.text(0., 1., 'Test', ha='left', va='top', bbox=props)
            ax.set_title('ELBOs of Forecasting on the Test Set')
            pp.savefig(fig)
            print ('ELBOs of forecasting on the test set plotted in {:.2f} seconds'.format(time.perf_counter()-time_start))
            time_start = time.perf_counter()
        if self.r_gt_all_test is not None:
            fig, ax = self.plot_firing_rates(self.r_gt_all_test)
            fig.text(0., 1., 'Test', ha='left', va='top', bbox=props)
            ax.set_title('Ground Truth Firing Rates')
            pp.savefig(fig)
            print ('Ground truth firing rates for the test set plotted in {:.2f} seconds'.format(time.perf_counter()-time_start))
            time_start = time.perf_counter()
        if self.r_pred_test is not None:
            fig, ax = self.plot_firing_rates(self.r_pred_test)
            fig.text(0., 1., 'Test', ha='left', va='top', bbox=props)
            ax.set_title('Predicted Firing Rates')
            pp.savefig(fig)
            print ('Prediction firing rates for the test set plotted in {:.2f} seconds'.format(time.perf_counter()-time_start))
            time_start = time.perf_counter()
        if self.r_gt_obs_test is not None:
            fig, ax = self.plot_distribution(self.r_gt_obs_test)
            fig.text(0., 1., 'Test', ha='left', va='top', bbox=props)
            ax.set_title('Observed Ground Truth Distribution')
            pp.savefig(fig)
            print ('Ground truth distribution for the test set plotted in {:.2f} seconds'.format(time.perf_counter()-time_start))
            time_start = time.perf_counter()
            if self.r_pred_test is not None:
                fig, ax_gt, ax_pred = self.plot_firing_rates_comparison(self.r_gt_obs_test, self.r_pred_test)
                fig.text(0., 1., 'Test', ha='left', va='top', bbox=props)
                pp.savefig(fig)
                print ('Firing rates comparison for the test set plotted in {:.2f} seconds'.format(time.perf_counter()-time_start))
                time_start = time.perf_counter()
                fig, ax = self.plot_residuals(self.r_gt_obs_test, self.r_pred_test, True)
                fig.text(0., 1., 'Test', ha='left', va='top', bbox=props)
                pp.savefig(fig)
                print ('Normalized residuals for the test set plotted in {:.2f} seconds'.format(time.perf_counter()-time_start))
                time_start = time.perf_counter()
                fig, ax = self.plot_residuals(self.r_gt_obs_test, self.r_pred_test, False)
                fig.text(0., 1., 'Test', ha='left', va='top', bbox=props)
                pp.savefig(fig)
                print ('Unnormalized residuals for the test set plotted in {:.2f} seconds'.format(time.perf_counter()-time_start))
                time_start = time.perf_counter()
                fig, ax = self.plot_neurons_comparison(self.r_gt_obs_test, self.r_pred_test)
                fig.text(0., 1., 'Test', ha='left', va='top', bbox=props)
                pp.savefig(fig)
                print ('Neurons comparison for the test set plotted in {:.2f} seconds'.format(time.perf_counter()-time_start))
                time_start = time.perf_counter()
        pp.close()

    def plot_firing_rates_comparison(self, r_gt, r_pred):
        '''
        Plot the observed ground truth firing rates compared to the predicted firing rates.
        '''
        assert r_gt is not None
        assert r_pred is not None

        inds_all = utils.get_ranked_indices(r_gt, thr=-1.5)
        n_neurons = r_gt.shape[1]
        T_temp = r_gt.shape[0]
        r_gt = r_gt[:,inds_all]
        r_pred = r_pred[:,inds_all]
        vmin, vmax = r_gt.min(), r_gt.max()

        fig  = plt.figure()
        ax_gt = fig.add_subplot(121)
        im = ax_gt.imshow(r_gt.T, aspect='auto', cmap='jet', vmin=vmin, vmax=vmax, extent=[-0.5, T_temp-0.5, n_neurons-0.5, -0.5], interpolation='none')
        ax_gt.set_xlabel('Time (step)')
        ax_gt.set_ylabel('Neuron rank')
        ax_gt.set_title('Ground truth')
        fig.colorbar(im)
        ax_pred = fig.add_subplot(122)
        im = ax_pred.imshow(r_pred.T, aspect='auto', cmap='jet', vmin=vmin, vmax=vmax, extent=[-0.5, T_temp-0.5, n_neurons-0.5, -0.5], interpolation='none')
        ax_pred.set_xlabel('Time (step)')
        ax_pred.set_ylabel('Neuron rank')
        ax_pred.set_title('Prediction')
        fig.colorbar(im)
        fig.tight_layout()
        return fig, ax_gt, ax_pred

    def plot_distribution(self, r):
        '''
        Plot the distribution of the ground truth firing rates.
        '''
        assert r is not None

        r_flat = r.flatten()
        fig, ax = plt.subplots()
        ax.hist(r_flat, bins=100)
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        return fig, ax

    def plot_neurons_comparison(self, r_gt, r_pred, indices = None):
        '''
        Compare the neural activities for the ground truth and the prediction for a subset of neurons.
        '''
        assert r_gt is not None
        assert r_pred is not None

        n_neurons = r_gt.shape[1]
        n_plotted = 10

        if indices is None:
            indices = np.random.choice(n_neurons, n_plotted, replace=False)
        r_gt = r_gt[:,indices]
        r_pred = r_pred[:,indices]
        lim = max(abs(r_gt).max(), abs(r_pred).max())

        fig, ax = plt.subplots()
        for i in range(n_plotted):
            ax.plot(r_gt[:,i] + lim * i, '-', label='Grount Truth' if i == 0 else None)
            ax.plot(r_pred[:,i] + lim * i, '--', label='Prediction' if i == 0 else None)
        ax.set_xlabel('Time (Step)')
        ax.set_yticks(np.arange(n_plotted) * lim, ["Neuron {}".format(i) for i in indices])
        ax.set_title('Select Neural Activities Comparison')
        ax.legend()
        fig.tight_layout()
        return fig, ax
    
    def plot_spectrum_comparison(self, eigenvals_gt, eigenvals_pred):
        '''
        Compare the eignevalues of the ground truth and the prediction.
        '''

        # Compare the eigenvalues of the student and the teacher
        fig, ax = plt.subplots()
        if eigenvals_gt is not None: ax.scatter(np.real(eigenvals_gt), np.imag(eigenvals_gt), color='red', marker='x', label='Ground Truth', linewidths=1)
        if eigenvals_pred is not None: ax.scatter(np.real(eigenvals_pred), np.imag(eigenvals_pred), color='blue', marker='o', facecolors='none', label='Learned')
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')

        theta = np.linspace(0, 2*np.pi, 100)
        x = np.cos(theta)
        y = np.sin(theta)
        ax.plot(x, y, color='gray', alpha=0.5, linestyle='-')

        ax.set_xlabel(r'$\Re(\lambda)$')
        ax.set_ylabel(r'$\Im(\lambda)$')
        ax.legend()
        return fig, ax
    
    def plot_dynamics(self, forecast, r_ex=None, r_init=None, flow_field=False):
        '''
        Visualize the trajectory of a trial and plot the flow field if applicable.
        Inputs:
        - forecast: Function to forecast (no inputs for now) with the following arguments:
            - r_in: initial firing rates (T_init, n)
            - T: number of total time steps
        - r_ex: an example trajectory (T, n)
        - r_init: initial firing rates (n_trajs, T_init, n)
        - flow_field: whether to plot the flow field
        - title: title of the plot
        '''

        # First, we generate the trajectories of some random starting points
        # For now, all r_ex and rs have to have the same T and n.
        n_trajs = 10 if r_init is None else r_init.shape[0]
        T = 30000 if r_ex is None else r_ex.shape[0]
        for i in range(n_trajs):
            r_in = r_init[i] if r_init is not None else None
            if i==0:
                r1 = forecast(T=T, r_in=r_in)
                n = r1.shape[1]
                rs = np.zeros((n_trajs, T, n))
            rs[i] = forecast(T=T, r_in=r_in)
        r_all = np.concatenate([r_ex[np.newaxis,...], rs], axis=0) if r_ex is not None else rs
        # First, we find the space spanned by the top two principal components
        # First, we do for the ground truth
        pca = PCA(n_components=2)
        pca.fit(r_all.reshape(-1, n))
        fig, ax = plt.subplots()
        colors = np.linspace(0, 1, T)
        if r_ex is not None:
            # Plot the example trajectory
            #TODO: Make trajectories with points so that the last point is on top.
            r_ex_xy = pca.transform(r_ex)
            lines = utils.colored_line(r_ex_xy[:,0], r_ex_xy[:,1], colors, ax, cmap=plt.cm.inferno)
            fig.colorbar(lines)  # add a color legend
        # Plot the other trajectories
        for i in range(n_trajs):
            r_xy = pca.transform(rs[i])
            lines = utils.colored_line(r_xy[:,0], r_xy[:,1], colors, ax, cmap=plt.cm.viridis)
        fig.colorbar(lines)
        ax.axis('equal')
        fig.canvas.draw()
        #ax.autoscale_view()

        if flow_field:
            # DOESN'T WORK YET
            # Plot vector field with color based on magnitude
            # Create meshgrid for vector field
            print (ax.get_ylim())
            x = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 20)
            y = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 20)
            X, Y = np.meshgrid(x, y)

            
            # Define vector components (U, V) - replace these with your actual data
            r_gts_now = pca.inverse_transform(np.stack((X.flatten(), Y.flatten()), axis=1)) # (length^2, n_rec)
            r_gts_next = rnn.forward_step(r_gts_now[...,np.newaxis,:]).squeeze() # (length^2, n_rec)
            vector_field = pca.transform(r_gts_next - r_gts_now) # (length^2, 2)

            U = vector_field[:,0].reshape(X.shape)
            V = vector_field[:,1].reshape(X.shape)

            # Calculate the magnitude of the vectors
            magnitude = np.sqrt(U**2 + V**2)

            ax.quiver(X, Y, U, V, angles='xy') # 'xy' means the direction always follows the data, not how the plot is shown
            ax.imshow(magnitude, cmap=plt.cm.gray, alpha=0.5, vmin=0, extent=(ax.axis()))
            #ax.axis('equal')
            #ax.autoscale_view()

        ax.set_xlabel('PC1 (A.U.)')
        ax.set_ylabel('PC2 (A.U.)')

        return fig, ax
    
    def plot_firing_rates(self, r):
        '''
        Plot the firing rates.
        '''
        assert r is not None

        # Plot the dynamics of the RNN
        inds_all = utils.get_ranked_indices(r, thr=-1.5)
        n_neurons = r.shape[1]
        T_temp = r.shape[0]
        r_sorted = r[:,inds_all]

        fig, ax = plt.subplots()
        im = ax.imshow(r_sorted.T, aspect='auto', cmap='jet', extent=[-0.5, T_temp-0.5, n_neurons-0.5, -0.5], interpolation='none')
        ax.set_xlabel('Time (step)')
        ax.set_ylabel('Neuron rank')
        fig.colorbar(im)
        fig.tight_layout()
        return fig, ax

    def plot_elbos(self, elbos):
        '''
        Plot the ELBOs.
        '''
        assert elbos is not None

        fig, ax = plt.subplots()
        ax.plot(elbos)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('ELBO')
        fig.tight_layout()
        return fig, ax

    def plot_residuals(self, r_gt, r_pred, norm = False):
        '''
        Plot the residuals of the prediction.
        Inputs:
        - norm: whether to normalize the residuals in each gt bin
        '''
        assert r_gt is not None
        assert r_pred is not None

        fig, ax = plt.subplots()
        num_bins_gt = 100
        num_bins_res = 100
        r_gt_flat = r_gt.flatten()
        r_pred_flat = r_pred.flatten()
        residuals = r_gt_flat - r_pred_flat
        h = ax.hist(r_gt_flat, num_bins_gt)
        ax.clear()

        if norm:
            out_bin_idx = np.digitize(r_gt_flat, h[1])-1
            out_bin_idx[np.where(out_bin_idx==num_bins_gt)] = num_bins_gt - 1
            out_weights = 1 / h[0][out_bin_idx]
            h = ax.hist2d(r_gt_flat, residuals, [num_bins_gt, num_bins_res], weights=out_weights, norm=colors.Normalize(vmin=0, vmax=0.1), cmap='Blues')
            ax.set_xlabel('Ground Truth')
            ax.set_ylabel('Residual (Gt - Prediction)')
            fig.colorbar(h[3], ax=ax, label='Fraction of Gt Bin')
            #graphs['ax_out_residual'].set_xlim(0,100)
            #graphs['ax_out_residual'].set_ylim(-2,2)
            ax.set_title('Residual plot normalized to each gt bin')

        else:
            h = ax.hist2d(r_gt_flat, residuals, [num_bins_gt, num_bins_res], norm=colors.LogNorm(), cmap='Blues')
            ax.set_xlabel('Ground Truth')
            ax.set_ylabel('Residual (Gt - Prediction)')
            fig.colorbar(h[3], ax=ax, label='Frequency')
            #graphs['ax_out_residual'].set_xlim(0,100)
            #graphs['ax_out_residual'].set_ylim(-2,2)
            ax.set_title('Residual plot')
            
        fig.tight_layout()
        return fig, ax