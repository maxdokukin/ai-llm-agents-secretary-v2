import argparse
import subprocess
import shlex
import sys


class LLMServer:
    """
    An encapsulated class to manage and start a llama.cpp server process.
    """

    def __init__(self, model: str, parameters: str, port: int):
        self.model = model
        self.parameters = parameters
        self.port = port
        self.process = None

    def start(self):
        """
        Constructs the command and starts the llama.cpp server.
        Assumes 'llama-server' is compiled and available in your system's PATH.
        """
        # Base command
        command = [
            "llama-server",
            "--model", self.model,
            "--port", str(self.port)
        ]

        # Safely parse and append additional parameters
        if self.parameters:
            try:
                extra_args = shlex.split(self.parameters)
                command.extend(extra_args)
            except ValueError as e:
                print(f"Error parsing parameters: {e}")
                sys.exit(1)

        print(f"Starting LLM Server...\nCommand: {' '.join(command)}\n")

        try:
            # Start the server process
            self.process = subprocess.Popen(command)
            # Block and wait for the process to finish
            self.process.wait()

        except FileNotFoundError:
            print("Error: 'llama-server' executable not found.")
            print(
                "Please ensure llama.cpp is compiled and 'llama-server' is in your system PATH, or provide the absolute path to the executable in the code.")
            sys.exit(1)
        except KeyboardInterrupt:
            print("\nReceived keyboard interrupt. Shutting down the LLM server safely...")
            self.stop()

    def stop(self):
        """Terminates the server process if it is running."""
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.process.wait()
            print("Server successfully stopped.")


if __name__ == "__main__":
    # The 4 Gemma-4 GGUF repositories requested
    SUPPORTED_MODELS = [
        "ggml-org/gemma-4-E2B-it-GGUF",
        "ggml-org/gemma-4-E4B-it-GGUF",
        "ggml-org/gemma-4-26B-A4B-it-GGUF",
        "ggml-org/gemma-4-31B-it-GGUF"
    ]

    parser = argparse.ArgumentParser(
        description="Start a llama.cpp server for Gemma-4 GGUF models.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help=(
                "Path to the downloaded .gguf file. Example models you might use this with:\n" +
                "\n".join([f"  - {m}" for m in SUPPORTED_MODELS])
        )
    )

    parser.add_argument(
        "--parameters",
        type=str,
        default="",
        help="Additional llama-server parameters as a single string. Example: '-c 8192 --n-gpu-layers 33'"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to run the server on (default: 8080)"
    )

    args = parser.parse_args()

    # Initialize and start the encapsulated server
    server = LLMServer(model=args.model, parameters=args.parameters, port=args.port)
    server.start()