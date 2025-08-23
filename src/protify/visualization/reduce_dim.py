import os
import numpy as np
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.manifold import TSNE as SklearnTSNE
from typing import Optional, Union, List
from matplotlib.colors import LinearSegmentedColormap
from utils import torch_load, print_message
from ..seed_utils import get_sklearn_random_state


@dataclass
class VisualizationArguments:
    embedding_save_dir: str = "embeddings"
    model_name: str = "ESM2-8"
    matrix_embed: bool = False
    sql: bool = False
    n_components: int = 2
    perplexity: float = 30.0  # for t-SNE
    n_neighbors: int = 15     # for UMAP
    min_dist: float = 0.1     # for UMAP
    seed: int = 42
    fig_size: tuple = (10, 10)
    save_fig: bool = True
    fig_dir: str = "figures"
    task_type: str = "singlelabel"  # singlelabel, multilabel, regression


class DimensionalityReducer:
    """Base class for dimensionality reduction techniques"""
    def __init__(self, args: VisualizationArguments):
        self.args = args
        self.embeddings = None
        self.labels = None
        
    def load_embeddings(self, sequences: List[str], labels: Optional[List[Union[int, float, List[int]]]] = None):
        """Load embeddings from file"""
        if self.args.sql:
            import sqlite3
            save_path = os.path.join(self.args.embedding_save_dir, 
                                   f'{self.args.model_name}_{self.args.matrix_embed}.db')
            embeddings = []
            with sqlite3.connect(save_path) as conn:
                c = conn.cursor()
                for seq in sequences:
                    c.execute("SELECT embedding FROM embeddings WHERE sequence = ?", (seq,))
                    embedding = c.fetchone()[0]
                    embedding = np.frombuffer(embedding, dtype=np.float32)
                    embeddings.append(embedding)
        else:
            save_path = os.path.join(self.args.embedding_save_dir,
                                   f'{self.args.model_name}_{self.args.matrix_embed}.pth')
            embeddings = []
            emb_dict = torch_load(save_path)
            for seq in sequences:
                embedding = emb_dict[seq].numpy()
                if self.args.matrix_embed:
                    embedding = embedding.mean(axis=0)
                embeddings.append(embedding)

        print_message(f"Loaded {len(embeddings)} embeddings")
        self.embeddings = np.stack(embeddings)
        if labels is not None:
            # Convert labels to a numpy array. For multi-label, this can be shape (num_samples, num_labels).
            self.labels = np.array(labels)
        else:
            self.labels = None
        
    def fit_transform(self):
        """Implement in child class"""
        raise NotImplementedError
        
    def plot(self, save_name: Optional[str] = None):
        """Plot the reduced dimensionality embeddings with appropriate coloring scheme"""
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call load_embeddings() first.")
            
        print_message("Fitting and transforming")
        reduced = self.fit_transform()
        print_message("Plotting")
        plt.figure(figsize=self.args.fig_size)
        
        if self.labels is None:
            # No labels - just a single color
            scatter = plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.6)
            
        elif self.args.task_type == "singlelabel":
            unique_labels = np.unique(self.labels)
            # Handle binary or multiclass
            if len(unique_labels) == 2:  # Binary classification
                colors = ['#ff7f0e', '#1f77b4']  # Orange and Blue
                cmap = LinearSegmentedColormap.from_list('binary', colors)
                scatter = plt.scatter(reduced[:, 0], reduced[:, 1], 
                                      c=self.labels, cmap=cmap, alpha=0.6)
                plt.colorbar(scatter, ticks=[0, 1])
            else:  # Multiclass classification
                n_classes = len(unique_labels)
                if n_classes <= 10:
                    cmap = 'tab10'
                elif n_classes <= 20:
                    cmap = 'tab20'
                else:
                    # For many classes, create a custom colormap
                    colors = sns.color_palette('husl', n_colors=n_classes)
                    cmap = LinearSegmentedColormap.from_list('custom', colors)
                
                scatter = plt.scatter(reduced[:, 0], reduced[:, 1], 
                                      c=self.labels, cmap=cmap, alpha=0.6)
                plt.colorbar(scatter, ticks=unique_labels)
                
        elif self.args.task_type == "multilabel":
            # For multi-label, create spectrum from blue to red along the label axis
            # where more blue if the labels are closer to index 0 and more red if the labels are closer to index -1
            # If there are more than one postive (multi-hot), average their colors
            label_colors = np.zeros(len(self.labels))
            label_counts = np.sum(self.labels, axis=1)
            
            # For samples with positive labels, calculate the weighted average position
            for i, label_row in enumerate(self.labels):
                if label_counts[i] > 0:
                    # Calculate weighted position (0 = first label, 1 = last label)
                    positive_indices = np.where(label_row == 1)[0]
                    avg_position = np.mean(positive_indices) / (self.labels.shape[1] - 1)
                    label_colors[i] = avg_position
                    
            # Create a blue to red colormap
            blue_red_cmap = LinearSegmentedColormap.from_list('blue_red', ['blue', 'red'])
            
            # Plot with both color dimensions: count and position
            scatter = plt.scatter(reduced[:, 0], reduced[:, 1], 
                                  c=label_colors, cmap=blue_red_cmap, 
                                  s=30 + 20 * label_counts, alpha=0.6)
            
            # Add two colorbars
            plt.colorbar(scatter, label='Label Position (blue=first, red=last)')
            
            # Add a size legend for count
            handles, labels = [], []
            for count in sorted(set(label_counts)):
                handles.append(plt.scatter([], [], s=30 + 20 * count, color='gray'))
                labels.append(f'{int(count)} labels')
            plt.legend(handles, labels, title='Label Count', loc='upper right')
            
        elif self.args.task_type == "regression":
            # For regression, use a sequential colormap
            vmin, vmax = np.percentile(self.labels, [2, 98])  # Robust scaling
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            scatter = plt.scatter(reduced[:, 0], reduced[:, 1], 
                                  c=self.labels, cmap='viridis', 
                                  norm=norm, alpha=0.6)
            plt.colorbar(scatter, label='Value')
        
        plt.title(f'{self.__class__.__name__} visualization of {self.args.model_name} embeddings')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        
        if save_name is not None and self.args.save_fig:
            os.makedirs(self.args.fig_dir, exist_ok=True)
            plt.savefig(os.path.join(self.args.fig_dir, save_name), 
                        dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()


class PCA(DimensionalityReducer):
    def __init__(self, args: VisualizationArguments):
        super().__init__(args)
        self.pca = SklearnPCA(n_components=args.n_components, random_state=get_sklearn_random_state() or args.seed)
        
    def fit_transform(self):
        return self.pca.fit_transform(self.embeddings)


class TSNE(DimensionalityReducer):
    def __init__(self, args: VisualizationArguments):
        super().__init__(args)
        self.tsne = SklearnTSNE(
            n_components=self.args.n_components,
            perplexity=self.args.perplexity,
            random_state=get_sklearn_random_state() or self.args.seed
        )
        
    def fit_transform(self):
        return self.tsne.fit_transform(self.embeddings)


class UMAP(DimensionalityReducer):
    def __init__(self, args: VisualizationArguments):
        super().__init__(args)
        self.umap = umap.UMAP(
            n_components=self.args.n_components,
            n_neighbors=self.args.n_neighbors,
            min_dist=self.args.min_dist,
            random_state=get_sklearn_random_state() or self.args.seed
        )
        
    def fit_transform(self):
        return self.umap.fit_transform(self.embeddings)


if __name__ == "__main__":
    # py -m visualization.reduce_dim
    ### TODO update with datamixin
    from data.hf_data import HFDataArguments, get_hf_data
    
    # Get some example data
    data_args = HFDataArguments(data_paths=["EC"])
    datasets, all_seqs = get_hf_data(data_args)
    
    # Get sequences and labels from first dataset
    dataset_name = list(datasets.keys())[0]
    train_set = datasets[dataset_name][0]
    sequences = train_set["seqs"]
    labels = train_set["labels"]  # Could be single label, multi-label, etc.
    
    # If you know your dataset is multi-label, specify it here
    vis_args = VisualizationArguments(
        embedding_save_dir="embeddings",
        model_name="ESMV",
        matrix_embed=False,
        sql=False,
        task_type="multilabel",  # Switch to 'multilabel'
        save_fig=True
    )
    
    for Reducer in [PCA, TSNE, UMAP]:
        print_message(f"Running {Reducer.__name__}")
        reducer = Reducer(vis_args)
        print_message("Loading embeddings")
        reducer.load_embeddings(sequences, labels)
        reducer.plot(f"{dataset_name}_{Reducer.__name__}.png")
