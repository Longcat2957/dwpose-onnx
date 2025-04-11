import logging
from typing import Tuple, List, Dict, Any, Optional
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from abc import ABC, abstractmethod


class ONNXWrapper(ABC):
    """
    Base class for wrapping ONNX models.
    This class provides functionality for loading and using ONNX models
    with proper device configuration and logging.
    """

    def __init__(
        self,
        device: str = "cuda",
        verbose: bool = False,
        repo_id: Optional[str] = None,
        filename: Optional[str] = None,
        model_path: Optional[str] = None,
    ) -> None:
        """
        Initialize the ONNX model wrapper.

        Args:
            device (str): Target device ('cpu', 'cuda', or 'cuda:device_id'). Defaults to 'cuda'.
            verbose (bool): If True, set logging level to INFO. Otherwise WARNING. Defaults to False.
            repo_id (str, optional): Hugging Face Hub repository ID. Required if using Hugging Face.
            filename (str, optional): Filename of the model in the repository. Required if using Hugging Face.
            model_path (str, optional): Direct path to the model file. Used if repo_id and filename are not provided.
        """
        # Set up logger
        self.logger = self._setup_logger(verbose)
        self.logger.info(f"Initializing {self.__class__.__name__}")

        # Download or load model
        self.model_path = self._get_model_path(repo_id, filename, model_path)

        # Configure ONNX session
        providers, provider_options = self.check_device(device)

        # Create session
        self.session = ort.InferenceSession(
            self.model_path, providers=providers, provider_options=provider_options
        )

        # Get input and output names
        self.input_names = self.get_input_names()
        self.output_names = self.get_output_names()
        _, _, model_h, model_w = self.session.get_inputs()[0].shape
        self.model_input_shape = (model_w, model_h)

        self.logger.info(f"Model loaded successfully: {self.model_path}")
        self.logger.info(f"Input nodes: {self.input_names}")
        self.logger.info(f"Output nodes: {self.output_names}")
        self.logger.info(f"Model input shape: {self.model_input_shape}")
        self.logger.info(f"Available providers: {providers}")

    def _setup_logger(self, verbose: bool) -> logging.Logger:
        """
        Set up a logger for the wrapper.

        Args:
            verbose (bool): If True, set logging level to INFO. Otherwise WARNING.

        Returns:
            logging.Logger: Configured logger instance.
        """
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO if verbose else logging.WARNING)

        # Add handler if none exists
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _get_model_path(
        self, repo_id: Optional[str], filename: Optional[str], model_path: Optional[str]
    ) -> str:
        """
        Get the path to the ONNX model file, either from Hugging Face or from a local path.

        Args:
            repo_id (str, optional): Hugging Face Hub repository ID.
            filename (str, optional): Filename of the model in the repository.
            model_path (str, optional): Direct path to the model file.

        Returns:
            str: Path to the model file.

        Raises:
            ValueError: If neither (repo_id and filename) nor model_path is provided.
        """
        if model_path is not None:
            self.logger.info(f"Using model from local path: {model_path}")
            return model_path
        elif repo_id is not None and filename is not None:
            self.logger.info(
                f"Downloading model from Hugging Face: {repo_id}/{filename}"
            )
            return hf_hub_download(repo_id=repo_id, filename=filename)
        else:
            raise ValueError(
                "Either (repo_id and filename) or model_path must be provided"
            )

    def check_device(self, device: str) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Check if the device is available and return the appropriate ONNXRuntime providers.

        Args:
            device (str): Device string (e.g., 'cpu', 'cuda', 'cuda:1', etc.).

        Returns:
            Tuple[List[str], List[dict]]:
                - providers (List[str]): Provider names for onnxruntime.
                - provider_options (List[dict]): Additional provider-specific options.
        """
        available_providers = ort.get_available_providers()

        if device.lower() == "cpu":
            self.logger.info("Using CPU as requested")
            return ["CPUExecutionProvider"], [{}]

        # CUDA device check
        if "CUDAExecutionProvider" in available_providers:
            if ":" in device:
                try:
                    gpu_id = int(device.split(":")[1])
                    self.logger.info(f"CUDA is available and will use GPU ID: {gpu_id}")
                    return ["CUDAExecutionProvider", "CPUExecutionProvider"], [
                        {"device_id": gpu_id},
                        {},
                    ]
                except (ValueError, IndexError):
                    self.logger.warning(
                        f"Invalid GPU ID format in '{device}'. Using default GPU (0)."
                    )
                    return ["CUDAExecutionProvider", "CPUExecutionProvider"], [
                        {"device_id": 0},
                        {},
                    ]
            else:
                self.logger.info("CUDA is available and will use default GPU (0)")
                return ["CUDAExecutionProvider", "CPUExecutionProvider"], [
                    {"device_id": 0},
                    {},
                ]
        else:
            self.logger.warning("CUDA is not available. Falling back to CPU.")
            return ["CPUExecutionProvider"], [{}]

    def get_input_names(self) -> List[str]:
        """
        Get the names of input nodes in the ONNX model.

        Returns:
            List[str]: Names of input nodes.
        """
        return [input_node.name for input_node in self.session.get_inputs()]

    def get_output_names(self) -> List[str]:
        """
        Get the names of output nodes in the ONNX model.

        Returns:
            List[str]: Names of output nodes.
        """
        return [output_node.name for output_node in self.session.get_outputs()]

    def get_input_shape(self, input_name: Optional[str] = None) -> Tuple[int, ...]:
        """
        Get the shape of an input node.

        Args:
            input_name (str, optional): Name of the input node. If None, returns the shape of the first input node.

        Returns:
            Tuple[int, ...]: Shape of the input node.

        Raises:
            ValueError: If the specified input_name does not exist.
        """
        if input_name is None:
            return self.session.get_inputs()[0].shape

        for input_node in self.session.get_inputs():
            if input_node.name == input_name:
                return input_node.shape

        raise ValueError(f"Input name '{input_name}' not found in model")

    @abstractmethod
    def infer(self, *args, **kwargs):
        """
        Abstract method that must be implemented by subclasses.
        This method should implement the specific inference logic for each model type.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Any: Model output. The specific type depends on the implementation.

        Raises:
            NotImplementedError: This is an abstract method.
        """
        raise NotImplementedError("Subclasses must implement infer()")

    def __call__(self, *args, **kwargs):
        """
        Make the wrapper callable. This is a convenience method that wraps the infer method.

        Args:
            *args: Variable length argument list passed to infer.
            **kwargs: Arbitrary keyword arguments passed to infer.

        Returns:
            Any: The result of calling infer.
        """
        self.logger.info(f"Calling {self.__class__.__name__}")
        return self.infer(*args, **kwargs)
