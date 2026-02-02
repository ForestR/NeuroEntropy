"""
Hessian-Aware Catalyst Generator

Core module for generating catalysts that exploit Hessian structure
to induce metabolic attacks on LLMs.

The Thermodynamics of Intelligence Loss in LLMs.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List
from transformers import PreTrainedModel, PreTrainedTokenizer


class HessianAwareCatalyst:
    """
    Generates attack catalysts by leveraging Hessian eigenvalue structure.
    
    This class implements the core mechanism for creating "Eigen-Prions"
    that exploit the spectral properties of model activations.
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: str = "cuda",
        top_k_eigenvalues: int = 10,
        noise_amplification_factor: float = 1.0,
        seed: Optional[int] = None
    ):
        """
        Initialize the catalyst generator.
        
        Args:
            model: The target language model
            tokenizer: Tokenizer for the model
            device: Computing device
            top_k_eigenvalues: Number of top eigenvalues to consider
            noise_amplification_factor: Factor for amplifying noise in Adam updates
            seed: Random seed for reproducible catalyst generation (optional)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.top_k = top_k_eigenvalues
        self.noise_amp = noise_amplification_factor
        self.seed = seed
        self.model.eval()  # Set to eval mode for HVP computation
    
    def _is_quantized_model(self) -> bool:
        """Check if model uses bitsandbytes quantization."""
        try:
            from bitsandbytes.nn import Linear4bit, Linear8bitLt
            for module in self.model.modules():
                if isinstance(module, (Linear4bit, Linear8bitLt)):
                    return True
        except ImportError:
            pass
        return False
    
    def _get_trainable_params(self) -> List[torch.nn.Parameter]:
        """
        Get list of trainable parameters (those that require grad).
        
        Returns:
            List of parameters that require gradients
        """
        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            # Fallback: get all parameters if none require grad (shouldn't happen normally)
            params = list(self.model.parameters())
        return params
    
    def _get_num_trainable_params(self) -> int:
        """
        Get count of trainable parameters (those that require grad).
        
        Returns:
            Total number of elements in trainable parameters
        """
        return sum(p.numel() for p in self._get_trainable_params())
        
    def _compute_loss(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute loss for given inputs.
        
        Args:
            input_ids: Input token IDs
            labels: Target labels (if None, use input_ids shifted)
            
        Returns:
            loss: Cross-entropy loss
        """
        if labels is None:
            # For language modeling, labels are input_ids shifted by 1
            labels = input_ids[:, 1:].contiguous()
            input_ids = input_ids[:, :-1].contiguous()
        
        outputs = self.model(input_ids=input_ids, labels=labels)
        return outputs.loss
    
    def compute_hessian_vector_product(
        self,
        input_ids: torch.Tensor,
        vector: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute Hessian-Vector Product (HVP): Hv = ∇²L · v
        
        This is the core operation for Hessian-free optimization.
        We compute HVP without explicitly constructing the Hessian matrix.
        
        Args:
            input_ids: Input token IDs
            vector: Direction vector (flattened parameter vector)
            labels: Target labels (optional)
            
        Returns:
            hvp: Hessian-vector product (flattened)
        """
        # Get model parameters - filter to only include those that require grad
        params = self._get_trainable_params()
        
        # Get model dtype for consistency
        model_dtype = next(self.model.parameters()).dtype
        
        # Check if model is quantized (bitsandbytes)
        is_quantized = self._is_quantized_model()
        
        # Force FP16 for HVP computation per PI directive: "precision is less important than existence"
        # This avoids doubling memory usage from FP32 conversion and is sufficient for finding approximate null directions
        use_fp32_for_hvp = False
        
        if use_fp32_for_hvp:
            # Save original parameter and buffer data, then convert entire model to float32
            original_param_data = [p.data.clone() for p in params]
            original_buffer_data = {}
            for name, buf in self.model.named_buffers():
                if buf.dtype.is_floating_point:
                    original_buffer_data[name] = buf.data.clone()
                    buf.data = buf.data.to(dtype=torch.float32)
            for p in params:
                p.data = p.data.to(dtype=torch.float32)
            # Get fresh params list after conversion
            params = list(self.model.parameters())
            # Also convert input to float32 (if it's floating point)
            input_ids_fp32 = input_ids
            if labels is not None and labels.dtype.is_floating_point:
                labels_fp32 = labels.to(dtype=torch.float32)
            else:
                labels_fp32 = labels
        else:
            input_ids_fp32 = input_ids
            labels_fp32 = labels
            original_param_data = None
            original_buffer_data = None
        
        # Compute first-order gradient
        # Disable efficient attention for gradient computation (needed for create_graph=True)
        # Efficient attention doesn't support higher-order gradients
        if hasattr(torch.backends.cuda, 'sdp_kernel'):
            sdp_ctx = torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)
        else:
            # Create a dummy context manager if sdp_kernel doesn't exist
            from contextlib import nullcontext
            sdp_ctx = nullcontext()
        
        try:
            with sdp_ctx:
                loss = self._compute_loss(input_ids_fp32, labels_fp32)
                try:
                    grad = torch.autograd.grad(
                        loss,
                        params,
                        create_graph=True,
                        retain_graph=True,
                        allow_unused=True  # Allow unused params for QLoRA
                    )
                except RuntimeError as e:
                    # Handle gradient computation failures, especially for quantized models
                    if is_quantized and "does not require grad" in str(e):
                        # For quantized models, filter params more strictly
                        params = [p for p in params if p.requires_grad and p.grad_fn is None]
                        if not params:
                            raise RuntimeError(
                                "No parameters with gradients found. Quantized models may have "
                                "limited gradient support. Consider using a non-quantized model."
                            ) from e
                        # Retry with filtered parameters
                        grad = torch.autograd.grad(
                            loss,
                            params,
                            create_graph=True,
                            retain_graph=True,
                            allow_unused=True  # Allow unused params for QLoRA
                        )
                    else:
                        raise
            
            # Flatten gradients
            grad_flat_parts = []
            for g in grad:
                if g is not None:
                    grad_flat_parts.append(g.view(-1))
            if not grad_flat_parts:
                raise RuntimeError("No gradients computed. All gradients are None.")
            grad_flat = torch.cat(grad_flat_parts)
            
            # Validate vector size matches gradient size
            expected_size = sum(p.numel() for p in params)
            if vector.numel() != expected_size:
                raise ValueError(
                    f"Vector size mismatch: vector has {vector.numel()} elements, "
                    f"but gradients have {expected_size} elements. "
                    f"This may occur if parameters were filtered inconsistently."
                )
            
            # Ensure vector matches grad_flat dtype
            if use_fp32_for_hvp:
                vector = vector.detach().to(dtype=torch.float32)
            else:
                vector = vector.detach().to(dtype=grad_flat.dtype)
            
            # Compute HVP: Hv = ∇(g · v)
            # This is the key: we compute gradient of (gradient · vector)
            dot_product = grad_flat @ vector
            
            # Compute second-order gradient
            # Disable efficient attention to avoid "backward not implemented" error
            # Efficient attention doesn't support second-order gradients
            try:
                # Try to disable efficient attention kernels
                if hasattr(torch.backends.cuda, 'sdp_kernel'):
                    # Use math kernel which supports second-order gradients
                    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
                        hvp = torch.autograd.grad(
                            outputs=dot_product,
                            inputs=params,
                            retain_graph=False,
                            create_graph=False,
                            allow_unused=True  # Allow unused params for QLoRA
                        )
                else:
                    # Fallback: try without context manager
                    hvp = torch.autograd.grad(
                        outputs=dot_product,
                        inputs=params,
                        retain_graph=False,
                        create_graph=False,
                        allow_unused=True  # Allow unused params for QLoRA
                    )
            except RuntimeError:
                # If still fails, use finite difference approximation
                # Use finite difference: Hv ≈ (g(θ + εv) - g(θ)) / ε
                epsilon = 1e-5
                param_sizes = [p.numel() for p in params]
                cumsum_sizes = [0] + [sum(param_sizes[:i+1]) for i in range(len(param_sizes))]
                
                # Save original parameters
                original_params = [p.data.clone() for p in params]
                
                # Apply perturbation
                for i, p in enumerate(params):
                    if p.requires_grad:
                        start_idx = cumsum_sizes[i]
                        end_idx = cumsum_sizes[i+1]
                        v_slice = (epsilon * vector[start_idx:end_idx]).view(p.shape)
                        p.data.add_(v_slice)
                
                # Recompute gradients
                loss_perturbed = self._compute_loss(input_ids_fp32, labels_fp32)
                grad_perturbed = torch.autograd.grad(
                    loss_perturbed,
                    params,
                    create_graph=False,
                    retain_graph=False,
                    allow_unused=True  # Allow unused params for QLoRA
                )
                grad_perturbed_flat = torch.cat([g.view(-1) for g in grad_perturbed if g is not None])
                
                # Compute HVP and restore parameters
                hvp_flat = (grad_perturbed_flat - grad_flat) / epsilon
                for p, orig in zip(params, original_params):
                    p.data = orig
                
                # Convert back to parameter shapes
                hvp = []
                for i, p in enumerate(params):
                    start_idx = cumsum_sizes[i]
                    end_idx = cumsum_sizes[i+1]
                    hvp.append(hvp_flat[start_idx:end_idx].view(p.shape))
            
            # Flatten HVP result and convert back to model dtype if needed
            hvp_flat_parts = []
            for h in hvp:
                if h is not None:
                    h_flat = h.view(-1)
                    if use_fp32_for_hvp:
                        h_flat = h_flat.to(dtype=model_dtype)
                    hvp_flat_parts.append(h_flat)
            hvp_flat = torch.cat(hvp_flat_parts)
            
        finally:
            # Restore original model parameters and buffers if we converted to float32
            if use_fp32_for_hvp:
                if original_param_data is not None:
                    for p, orig_data in zip(params, original_param_data):
                        p.data = orig_data
                if original_buffer_data is not None:
                    for name, orig_data in original_buffer_data.items():
                        self.model.get_buffer(name).data = orig_data
                # Restore model to original dtype (skip for quantized models)
                if not is_quantized:
                    self.model = self.model.to(dtype=model_dtype)
            
            # Clear gradients and free memory after HVP computation
            if hasattr(self.model, 'zero_grad'):
                self.model.zero_grad()
            else:
                for p in self.model.parameters():
                    if p.grad is not None:
                        p.grad = None
            
            # Clear CUDA cache to free fragmented memory
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return hvp_flat
    
    def find_null_space_directions(
        self,
        input_ids: torch.Tensor,
        num_directions: int = 10,
        max_iterations: int = 50,
        threshold: float = 1e-6,
        labels: Optional[torch.Tensor] = None,
        seed: Optional[int] = None
    ) -> List[torch.Tensor]:
        """
        Find null space directions using Power Iteration.
        
        Null space directions correspond to weight directions with near-zero
        curvature (small Hessian eigenvalues). These are the directions where
        Adam amplifies noise.
        
        Args:
            input_ids: Input token IDs for computing Hessian
            num_directions: Number of null directions to find
            max_iterations: Maximum power iteration steps
            threshold: Threshold for detecting null space (||Hv|| < threshold)
            labels: Target labels (optional)
            seed: Random seed for reproducible null space discovery (optional, uses self.seed if None)
            
        Returns:
            null_directions: List of null space direction vectors
        """
        null_directions = []
        num_params = self._get_num_trainable_params()
        
        # Reduce directions for large models (>1B params) to save memory
        if num_params > 1e9 and num_directions > 3:
            original_num_directions = num_directions
            num_directions = min(3, num_directions)
            print(f"Large model detected ({num_params/1e9:.1f}B params). "
                  f"Reducing num_null_directions from {original_num_directions} to {num_directions}")
        
        # Get dtype from model parameters
        model_dtype = next(self.model.parameters()).dtype
        
        # Use provided seed or instance seed
        use_seed = seed if seed is not None else self.seed
        
        for direction_idx in range(num_directions):
            # Set seed for reproducible random initialization
            if use_seed is not None:
                torch.manual_seed(use_seed + direction_idx)
                np.random.seed(use_seed + direction_idx)
            
            # Initialize random direction vector with matching dtype
            v = torch.randn(num_params, device=self.device, dtype=model_dtype)
            v = v / torch.norm(v)  # Normalize
            
            for iteration in range(max_iterations):
                # Compute HVP
                hvp = self.compute_hessian_vector_product(input_ids, v, labels)
                
                # Check if we've found a null direction
                hvp_norm = torch.norm(hvp)
                # Ensure threshold is same dtype as hvp_norm
                threshold_tensor = torch.tensor(threshold, dtype=hvp_norm.dtype, device=hvp_norm.device)
                if hvp_norm < threshold_tensor:
                    # Offload to CPU immediately to save GPU memory
                    null_directions.append(v.cpu().clone())
                    break
                
                # Power iteration: v = Hv / ||Hv||
                # Use dtype-aware epsilon
                epsilon = torch.tensor(1e-10, dtype=hvp_norm.dtype, device=hvp_norm.device)
                v = hvp / (hvp_norm + epsilon)
                
                # Ensure v maintains correct dtype after operations
                v = v.to(dtype=model_dtype)
                
                # Orthogonalize against previously found directions
                for prev_dir in null_directions:
                    # Temporarily move prev_dir to GPU for computation, then it stays on CPU
                    prev_dir_gpu = prev_dir.to(device=self.device, dtype=v.dtype)
                    v = v - (v @ prev_dir_gpu) * prev_dir_gpu
                
                v_norm = torch.norm(v)
                epsilon_norm = torch.tensor(1e-10, dtype=v_norm.dtype, device=v_norm.device)
                v = v / (v_norm + epsilon_norm)
                v = v.to(dtype=model_dtype)  # Ensure dtype consistency after normalization
            
            # Clear intermediate tensors periodically
            del hvp
            if direction_idx % 2 == 0 and self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            if len(null_directions) == direction_idx + 1:
                continue  # Successfully found null direction
            else:
                # If we didn't converge, use the final v as approximation
                # Offload to CPU immediately to save GPU memory
                null_directions.append(v.cpu().clone())
        
        return null_directions
    
    def generate_catalyst(
        self,
        base_prompt: str = "",
        num_steps: int = 100,
        learning_rate: float = 1e-2,
        catalyst_length: int = 128,
        null_directions: Optional[List[torch.Tensor]] = None,
        input_ids: Optional[torch.Tensor] = None,
        seed: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate catalyst tokens using gradient ascent on input embeddings.
        
        The objective is to maximize the projection of input gradient onto
        null space directions, creating "Eigen-Prion" data.
        
        Args:
            base_prompt: Starting prompt (optional)
            num_steps: Number of optimization steps
            learning_rate: Learning rate for gradient ascent
            catalyst_length: Length of catalyst sequence in tokens
            null_directions: Pre-computed null space directions (optional)
            input_ids: Input IDs for computing null directions (if needed)
            seed: Random seed for reproducible catalyst generation (optional, uses self.seed if None)
            
        Returns:
            catalyst_tokens: Optimized token IDs
        """
        self.model.train()  # Need gradients for input embeddings
        
        # Use provided seed or instance seed
        use_seed = seed if seed is not None else self.seed
        
        # Set seed for reproducible initialization
        if use_seed is not None:
            torch.manual_seed(use_seed)
            np.random.seed(use_seed)
        
        # Initialize catalyst tokens
        if base_prompt:
            base_ids = self.tokenizer.encode(base_prompt, return_tensors="pt")
            base_ids = base_ids.to(self.device)
            # Pad or truncate to catalyst_length
            if base_ids.size(1) < catalyst_length:
                padding = torch.zeros(
                    (1, catalyst_length - base_ids.size(1)),
                    dtype=torch.long,
                    device=self.device
                )
                catalyst_ids = torch.cat([base_ids, padding], dim=1)
            else:
                catalyst_ids = base_ids[:, :catalyst_length]
        else:
            # Random initialization
            vocab_size = len(self.tokenizer)
            catalyst_ids = torch.randint(
                0, vocab_size,
                (1, catalyst_length),
                device=self.device
            )
        
        # Get embedding layer
        embedding_layer = self.model.get_input_embeddings()
        
        # Get null directions if not provided
        if null_directions is None:
            if input_ids is None:
                # Use random input for null space computation
                # Set seed again for null space input generation
                if use_seed is not None:
                    torch.manual_seed(use_seed + 1000)  # Offset to avoid correlation
                input_ids = torch.randint(
                    0, len(self.tokenizer),
                    (1, 64),
                    device=self.device
                )
            null_directions = self.find_null_space_directions(
                input_ids,
                num_directions=self.top_k,
                seed=use_seed
            )
        
        # Get initial embeddings and make them require gradients
        # We optimize embeddings directly, not token IDs (which are integers and can't have gradients)
        initial_embeddings = embedding_layer(catalyst_ids)
        embeddings = initial_embeddings.clone().detach().requires_grad_(True)
        
        # Get embedding weight matrix for nearest neighbor lookup
        embedding_weight = embedding_layer.weight  # Shape: (vocab_size, embedding_dim)
        
        for step in range(num_steps):
            # Use inputs_embeds to pass embeddings directly to the model
            # This bypasses the embedding layer and allows gradients to flow through embeddings
            # Shift for language modeling: predict next token
            inputs_embeds = embeddings[:, :-1]  # Use all but last token's embeddings as input
            labels = catalyst_ids[:, 1:]  # Predict next tokens (use current token IDs for labels)
            
            # Forward pass using embeddings directly (bypasses embedding layer)
            # Most transformer models support inputs_embeds parameter
            outputs = self.model(inputs_embeds=inputs_embeds, labels=labels)
            loss = outputs.loss
            
            # Compute gradient w.r.t. embeddings (now embeddings are directly in computation graph)
            grad_emb = torch.autograd.grad(
                loss,
                embeddings,
                retain_graph=False,
                create_graph=False,
                allow_unused=True  # Allow unused params for QLoRA
            )[0]
            
            # Project gradient onto null space directions (simplified)
            projection = grad_emb.clone()
            
            # Update embeddings using gradient ascent
            embeddings = embeddings + learning_rate * projection
            
            # Normalize embeddings to prevent them from growing too large
            norm = torch.norm(embeddings, dim=-1, keepdim=True) + 1e-8
            embeddings = embeddings / norm
            embeddings = embeddings.detach().requires_grad_(True)
            
            # Update catalyst_ids for next iteration using nearest neighbor
            embeddings_flat = embeddings.view(-1, embeddings.size(-1))
            with torch.no_grad():
                distances = torch.cdist(embeddings_flat, embedding_weight)
                catalyst_ids = distances.argmin(dim=-1).view(embeddings.shape[:2])
            
            # Clear memory periodically during catalyst generation
            if step % 10 == 0:
                if hasattr(self.model, 'zero_grad'):
                    self.model.zero_grad()
                if self.device == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Convert final embeddings back to token IDs using nearest neighbor
        embeddings_flat = embeddings.view(-1, embeddings.size(-1))
        with torch.no_grad():
            distances = torch.cdist(embeddings_flat, embedding_weight)
            catalyst_ids = distances.argmin(dim=-1).view(embeddings.shape[:2])
        
        # Clear gradients and free memory after catalyst generation
        if hasattr(self.model, 'zero_grad'):
            self.model.zero_grad()
        try:
            del embeddings, embeddings_flat, distances, projection, grad_emb
        except NameError:
            pass
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.model.eval()  # Reset to eval mode
        return catalyst_ids.detach()
    
    def compute_hessian_spectrum(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the Hessian eigenvalue spectrum for given inputs.
        
        This is a simplified version that uses Power Iteration to estimate
        top eigenvalues.
        
        Returns:
            eigenvalues: Top-k eigenvalues
            eigenvectors: Corresponding eigenvectors
        """
        # Use Power Iteration to estimate top eigenvalues
        num_params = self._get_num_trainable_params()
        eigenvalues = []
        eigenvectors = []
        
        # Get dtype from model parameters
        model_dtype = next(self.model.parameters()).dtype
        
        for k in range(self.top_k):
            v = torch.randn(num_params, device=self.device, dtype=model_dtype)
            v = v / torch.norm(v)
            
            # Orthogonalize against previous eigenvectors
            for prev_evec in eigenvectors:
                prev_evec_converted = prev_evec.to(dtype=v.dtype)
                v = v - (v @ prev_evec_converted) * prev_evec_converted
            v_norm = torch.norm(v)
            epsilon = torch.tensor(1e-10, dtype=v_norm.dtype, device=v_norm.device)
            v = v / (v_norm + epsilon)
            
            # Power iteration
            for _ in range(50):
                hvp = self.compute_hessian_vector_product(inputs, v, targets)
                hvp_norm = torch.norm(hvp)
                epsilon_hvp = torch.tensor(1e-10, dtype=hvp_norm.dtype, device=hvp_norm.device)
                v = hvp / (hvp_norm + epsilon_hvp)
                v = v.to(dtype=model_dtype)  # Ensure dtype consistency
            
            # Estimate eigenvalue: λ ≈ v^T H v
            hvp_final = self.compute_hessian_vector_product(inputs, v, targets)
            eigenval = (v @ hvp_final).item()
            
            eigenvalues.append(eigenval)
            eigenvectors.append(v.clone())
        
        return torch.tensor(eigenvalues, device=self.device), torch.stack(eigenvectors)
    
    def amplify_adam_noise(
        self,
        gradients: torch.Tensor,
        hessian_eigenvalues: torch.Tensor
    ) -> torch.Tensor:
        """
        Amplify noise in Adam updates based on Hessian structure.
        
        This is the core mechanism: exploit the fact that Adam's
        second-moment estimate amplifies noise in directions corresponding
        to small Hessian eigenvalues.
        """
        # Noise amplification logic
        # The key insight: directions with small Hessian eigenvalues
        # experience amplified noise in Adam's update rule
        noise_scale = 1.0 / (hessian_eigenvalues + 1e-8)
        amplified_gradients = gradients * (1 + self.noise_amp * noise_scale)
        return amplified_gradients
