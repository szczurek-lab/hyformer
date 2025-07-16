import os
import logging
import torch
from typing import Dict, Optional

from torch.utils.data import DataLoader, DistributedSampler

from hyformer.utils.data.dataloaders.base import BaseDataLoader, PAD_TO_MULTIPLE_OF
from hyformer.utils.data.collators import DataCollatorWithTaskTokens
from hyformer.utils.datasets.base import BaseDataset
from hyformer.utils.reproducibility import seed_worker

console = logging.getLogger(__name__)


class SequenceDataLoader(BaseDataLoader):
    """Concrete implementation for sequence-based data loading.
    
    This factory provides sequence-specific implementations for creating data loaders
    with configurations optimized for text/sequence processing tasks.
    """
    
    def create_collator(
        self, 
        tasks: Dict[str, float],
        pad_to_multiple_of: Optional[int] = None,
        max_length: Optional[int] = None,
        mask_probability: float = 0.15
    ) -> DataCollatorWithTaskTokens:
        """Create a data collator with the specified tasks.
        
        Parameters
        ----------
        tasks : Dict[str, float]
            Dictionary mapping task names to their weights
        pad_to_multiple_of : int, optional
            Pad sequences to multiple of this value for better GPU utilization,
            by default PAD_TO_MULTIPLE_OF
        max_length : int, optional
            Maximum sequence length, by default uses tokenizer.max_length
        mask_probability : float, optional
            Mask probability for MLM tasks, by default 0.15
            
        Returns
        -------
        DataCollatorWithTaskTokens
            Configured data collator
        """
        return DataCollatorWithTaskTokens(
            tokenizer=self.tokenizer,
            tasks=tasks,
            pad_to_multiple_of=pad_to_multiple_of or PAD_TO_MULTIPLE_OF,
            max_length=max_length or self.tokenizer.max_length,
            return_tensors="pt",
            mask_probability=mask_probability
        )
    
    def create_sampler(
        self, 
        dataset: BaseDataset,
        distributed: bool = None
    ) -> Optional[DistributedSampler]:
        """Create a distributed sampler if running in distributed mode.
        
        Parameters
        ----------
        dataset : BaseDataset
            Dataset to create sampler for
        distributed : bool, optional
            Whether to create a distributed sampler. If None, auto-detects
            from torch.distributed.is_initialized()
            
        Returns
        -------
        DistributedSampler or None
            DistributedSampler if distributed=True, None otherwise
        """
        if distributed is None:
            distributed = torch.distributed.is_initialized()
            
        if not distributed:
            return None
            
        return DistributedSampler(
            dataset,
            num_replicas=int(os.environ["WORLD_SIZE"]),
            rank=int(os.environ["SLURM_PROCID"])
        )
    
    def create_loader(
        self,
        dataset: Optional[BaseDataset],
        tasks: Dict[str, float],
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: Optional[int] = None,
        distributed: bool = None,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
        pad_to_multiple_of: Optional[int] = None,
        max_length: Optional[int] = None,
        mask_probability: float = 0.15,
        **dataloader_kwargs
    ) -> Optional[DataLoader]:
        """Create a data loader with optimized settings for sequence data.
        
        Parameters
        ----------
        dataset : BaseDataset, optional
            Dataset to create loader for. If None, returns None
        tasks : Dict[str, float]
            Dictionary of tasks to use for collation. 
            Supported tasks are 'lm', 'prediction', and 'mlm'
        batch_size : int, optional
            Batch size, by default 32
        shuffle : bool, optional
            Whether to shuffle the data, by default True
        num_workers : int, optional
            Number of workers for data loading, by default auto-detected
        distributed : bool, optional
            Whether to use distributed sampling, by default auto-detected
        pin_memory : bool, optional
            Whether to pin memory for faster GPU transfer, by default True
        persistent_workers : bool, optional
            Whether to keep workers alive between epochs, by default True
        prefetch_factor : int, optional
            Number of batches to prefetch per worker, by default 2
        pad_to_multiple_of : int, optional
            Pad sequences to multiple of this value, by default PAD_TO_MULTIPLE_OF
        max_length : int, optional
            Maximum sequence length, by default uses tokenizer.max_length
        mask_probability : float, optional
            Mask probability for MLM tasks, by default 0.15
        **dataloader_kwargs
            Additional arguments to pass to DataLoader
            
        Returns
        -------
        DataLoader or None
            Configured DataLoader instance or None if dataset is None
            
        Raises
        ------
        ValueError
            If any task in tasks is not supported
        """
        if dataset is None:
            return None
            
        # Validate tasks using base class method
        self.validate_tasks(tasks)
        
        try:
            # Create sequence-specific components
            collator = self.create_collator(
                tasks=tasks,
                pad_to_multiple_of=pad_to_multiple_of,
                max_length=max_length,
                mask_probability=mask_probability
            )
            sampler = self.create_sampler(dataset, distributed=distributed)
            
            # Use base class methods for common functionality
            num_workers = self.get_num_workers(num_workers)
            generator = self.create_generator()
            
            data_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle if sampler is None else False,
                collate_fn=collator,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers and num_workers > 0,
                prefetch_factor=prefetch_factor if num_workers > 0 else None,
                worker_init_fn=seed_worker if num_workers > 0 else None,
                generator=generator,
                **dataloader_kwargs
            )
            
            console.debug(f"Created data loader with {len(data_loader)} batches")
            return data_loader
        except Exception as e:
            console.error(f"Error creating data loader: {e}")
            raise 
        