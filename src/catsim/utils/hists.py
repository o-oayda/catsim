import os
import pickle
from typing import Optional
import numpy as np
from numpy.typing import NDArray


class MultinomialSample2DHistogram:
    """
    Fast 2D histogram sampling using multinomial distribution.
    
    This approach treats the 2D histogram as a single multinomial distribution
    over all bins, allowing for very fast sampling by directly using np.random.choice.
    Expected to be ~10-20x faster than the conditional CDF approach.
    """
    
    def __init__(self) -> None:
        pass

    def build(self,
            x_data = None,
            y_data = None,
            **hist_kwargs
        ) -> None:
        """
        Build the multinomial sampler from data with jittering support.
        
        Parameters:
        -----------
        x_data : array-like
            X coordinates of data points
        y_data : array-like  
            Y coordinates of data points
        **hist_kwargs : dict
            Additional arguments passed to np.histogram2d
        """
        
        # Create 2D histogram
        counts_2d, self.x_edges, self.y_edges = np.histogram2d(
            x_data, y_data, **hist_kwargs # pyright: ignore[reportArgumentType, reportCallIssue]
        )
        
        # Calculate bin centers
        x_centres = (self.x_edges[:-1] + self.x_edges[1:]) / 2
        y_centres = (self.y_edges[:-1] + self.y_edges[1:]) / 2
        
        # Calculate bin widths for jittering
        self.x_bin_widths = np.diff(self.x_edges)
        self.y_bin_widths = np.diff(self.y_edges)
        
        # Create 2D coordinate grids for centers and widths
        self.x_centres_2d, self.y_centres_2d = np.meshgrid(
            x_centres, y_centres, indexing='ij'
        )
        x_widths_2d, y_widths_2d = np.meshgrid(
            self.x_bin_widths, self.y_bin_widths, indexing='ij'
        )
        
        # Flatten coordinate grids for multinomial sampling
        self.x_flat = self.x_centres_2d.flatten()
        self.y_flat = self.y_centres_2d.flatten()
        self.x_widths_flat = x_widths_2d.flatten()
        self.y_widths_flat = y_widths_2d.flatten()
        
        # Flatten counts and normalize to probabilities
        counts_flat = counts_2d.flatten()
        self.probs_flat = counts_flat / np.sum(counts_flat)
        
        # Store original shape for potential debugging
        self.original_shape = counts_2d.shape
        
        # Filter out zero-probability bins for efficiency (optional)
        nonzero_mask = self.probs_flat > 0
        if np.sum(nonzero_mask) < len(self.probs_flat):
            self.x_flat = self.x_flat[nonzero_mask]
            self.y_flat = self.y_flat[nonzero_mask]
            self.x_widths_flat = self.x_widths_flat[nonzero_mask]
            self.y_widths_flat = self.y_widths_flat[nonzero_mask]
            self.probs_flat = self.probs_flat[nonzero_mask]
            # Renormalize after filtering
            self.probs_flat = self.probs_flat / np.sum(self.probs_flat)
        
        print(f"MultinomialSample2DHistogram built with {len(self.probs_flat)} active bins")

    def save_data(self, save_dir: str) -> None:
        """Save the multinomial sampler data."""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        sampler_data = {
            'x_flat': self.x_flat,
            'y_flat': self.y_flat,
            'x_widths_flat': self.x_widths_flat,
            'y_widths_flat': self.y_widths_flat,
            'probs_flat': self.probs_flat,
            'x_edges': self.x_edges,
            'y_edges': self.y_edges,
            'original_shape': self.original_shape
        }
        
        with open(f'{save_dir}multinomial_sampler_data.pkl', 'wb') as handle:
            pickle.dump(sampler_data, handle)

    def load_data(self, save_dir: str) -> None:
        """Load the multinomial sampler data."""
        with open(f'{save_dir}multinomial_sampler_data.pkl', 'rb') as handle:
            sampler_data = pickle.load(handle)
        
        self.x_flat = sampler_data['x_flat']
        self.y_flat = sampler_data['y_flat']
        self.x_widths_flat = sampler_data['x_widths_flat']
        self.y_widths_flat = sampler_data['y_widths_flat']
        self.probs_flat = sampler_data['probs_flat']
        self.x_edges = sampler_data['x_edges']
        self.y_edges = sampler_data['y_edges']
        self.original_shape = sampler_data['original_shape']

    def sample(
            self,
            n_samples: int,
            rng: Optional[np.random.Generator] = None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Sample from the 2D distribution using multinomial sampling with uniform jittering.
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
            
        Returns:
        --------
        x_samples : NDArray[np.float64]
            X coordinates of samples with uniform jittering within bins
        y_samples : NDArray[np.float64]
            Y coordinates of samples with uniform jittering within bins
        """
        if rng is None:
            rng = np.random.default_rng()

        # Multinomial sampling to select bins
        indices = rng.choice(
            len(self.probs_flat), 
            size=n_samples, 
            p=self.probs_flat
        )
        
        # Get bin centers and widths for selected bins
        x_centers = self.x_flat[indices]
        y_centers = self.y_flat[indices]
        x_widths = self.x_widths_flat[indices]
        y_widths = self.y_widths_flat[indices]
        
        # Add uniform jitter within each bin
        # Jitter is uniform in [-width/2, +width/2] around bin center
        x_jitter = rng.uniform(-0.5, 0.5, n_samples) * x_widths
        y_jitter = rng.uniform(-0.5, 0.5, n_samples) * y_widths
        
        # Apply jittering to get continuous samples
        x_samples = x_centers + x_jitter
        y_samples = y_centers + y_jitter
        
        return x_samples, y_samples
    
    def get_bin_info(self) -> dict:
        """
        Get information about the binning for debugging/analysis.
        
        Returns:
        --------
        info : dict
            Dictionary containing bin information
        """
        return {
            'n_bins_total': len(self.x_flat),
            'x_range': (self.x_edges[0], self.x_edges[-1]),
            'y_range': (self.y_edges[0], self.y_edges[-1]),
            'original_shape': self.original_shape,
            'min_probability': np.min(self.probs_flat),
            'max_probability': np.max(self.probs_flat)
        }
