import json
import tempfile
from pathlib import Path

import cog
from cog_crepe.core import *
from imageio import imwrite


class Predictor(cog.Predictor):
    def setup(self):
        """Load the model"""
        self.model = build_and_load_model("full")

    @cog.input("input", type=Path, help="Audio file")
    @cog.input(
        "viterbi",
        type=bool,
        default=False,
        help="Apply viterbi smoothing to the estimated pitch curve",
    )
    @cog.input(
        "plot_voicing",
        type=bool,
        default=False,
        help="Include a visual representation of the voicing activity detection",
    )
    @cog.input(
        "step_size",
        type=int,
        default=10,
        help="The step size in milliseconds for running pitch estimation",
    )
    @cog.input(
        "output_type",
        type=str,
        default="plot",
        options=["plot", "json"],
        help="Type of output representation: could be plot or json (list of [time, frequency, confidence] values)",
    )
    def predict(self, input, viterbi, plot_voicing, step_size, output_type):
        """Compute f0 plot"""
        output_path = Path(tempfile.mkdtemp()) / "output.png"
        plot, f0_data = process_file(
            str(input),
            self.model,
            save_plot=True,
            viterbi=viterbi,
            plot_voicing=plot_voicing,
            step_size=step_size,
        )

        if output_type == "plot":
            imwrite(output_path, plot)
            return output_path

        elif output_type == "json":
            out = f0_data.tolist()
            return json.dumps(out)
