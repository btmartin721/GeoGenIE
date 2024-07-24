from pathlib import Path

import torch
import torch.nn as nn
from torchviz import make_dot
from torch.utils.tensorboard import SummaryWriter


class MLPRegressor(nn.Module):
    """Define PyTorch MLP Model."""

    def __init__(
        self,
        input_size,
        width=256,
        nlayers=10,
        dropout_prop=0.25,
        device="cpu",
        output_width=2,
        dtype=torch.float32,
    ):
        super(MLPRegressor, self).__init__()
        self.device = device
        self.dtype = dtype

        self.seqmodel = self._define_model(
            input_size, width, nlayers, dropout_prop, output_width
        )

    def _define_model(self, input_size, width, nlayers, dropout_prop, output_width):
        layers = [
            nn.BatchNorm1d(input_size, dtype=self.dtype),
            nn.Linear(input_size, width, dtype=self.dtype),
            nn.ELU(),
        ]
        for _ in range(nlayers):
            layers.append(nn.Linear(width, width, dtype=self.dtype))
            layers.append(nn.ELU())
        layers.append(nn.Dropout(dropout_prop))
        layers.append(nn.Linear(width, output_width, dtype=self.dtype))
        return nn.Sequential(*layers)

    def forward(self, x):
        if x.size(0) == 1:
            return self.seqmodel[1:](x)
        else:
            return self.seqmodel(x)

    def plot_model_nn(self, model, input_size, file_path):
        # Set the model to inference mode
        model.eval()

        # Create a dummy input tensor
        dummy_input = torch.randn(256, input_size, requires_grad=True)

        onnx_pth = file_path.parent / "model.onnx"

        # Export the model to ONNX format
        torch.onnx.export(
            model,  # Model being run
            dummy_input,  # Model input (or a tuple for multiple inputs)
            onnx_pth,  # Where to save the model
            export_params=True,  # Store the trained parameter weights inside the model file
            opset_version=10,  # The ONNX version to export the model to
            do_constant_folding=True,  # Whether to execute constant folding for optimization
            input_names=["SNPs"],  # The model's input names
            output_names=["Geolocation Predictions"],  # The model's output names
            dynamic_axes={
                "SNPs": {0: "batch_size"},  # Variable length axes
                "Geolocation Predictions": {0: "batch_size"},
            },
        )

        print("Model has been converted to ONNX")

        y = model(dummy_input)
        dot = make_dot(y, params=dict(model.named_parameters()))
        dot.format = "pdf"

        pth = Path(*file_path.parts[:-2]) / "model_architecture"
        dot.render(pth)
        print(f"Simplified model architecture diagram saved as {pth}.pdf")


def save_model(model, model_name="modelarch_trained_model.pt", directory="models"):
    """Save the trained model to a specified directory.

    Args:
        model: The trained model object.
        model_name: The name of the file to save the model as.
        directory: The directory where the model will be saved.
    """
    path = Path.cwd() / "summarize_results" / directory
    path.mkdir(exist_ok=True, parents=True)
    file_path = path / model_name
    torch.save(model.state_dict(), file_path)
    return file_path


def train_step(input_size, model, writer):
    """Train Step for a PyTorch model.

    Args:
        input_size (int): Number of features in inputs.
        model (torch.)
    """
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(50):
        model.train()
        # Random inputs. just for plotting.
        dummy_input = torch.randn(32, input_size)  # Batch size of 32
        dummy_target = torch.randn(32, 2)  # Output size of 2
        opt.zero_grad()
        output = model(dummy_input)
        loss = criterion(output, dummy_target)
        writer.add_scalar("Loss/train", loss, epoch)
        loss.backward()
        opt.step()
    return model, dummy_input, writer


if __name__ == "__main__":
    # Define initial sizes
    input_size = 413
    nlayers = 2
    model = MLPRegressor(input_size=input_size, nlayers=nlayers)

    writer = SummaryWriter()

    # Dummy training loop just for plotting architecture.
    model, dummy_input, writer = train_step(input_size, model, writer)

    of = save_model(model)

    writer.add_graph(model, dummy_input)
    writer.close()

    model = MLPRegressor(input_size=input_size, nlayers=nlayers)

    # Load a trained model.
    model.load_state_dict(torch.load(of))

    # Convert model to ONNX format and make plot.
    model.plot_model_nn(model, input_size, of)
