def build_backend(args):
    provider = getattr(args, "provider", "vllm")
    if provider == "vllm":
        from .vllm_backend import VllmBackend
        return VllmBackend(
            model_path=args.model_path,
            gpu=args.gpu,
        )
    elif provider == "modal_workers":
        from .modal_workers import ModalWorkersBackend
        return ModalWorkersBackend(
            app_name=getattr(args, "app_name", "goedel-prover-modal-workers"),
            function_name=getattr(args, "function_name", "generate_n_for_prompt"),
            model_name=getattr(args, "model_path", None),
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")
