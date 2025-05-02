from modules.module_base import ModuleBase
from PIL import Image
from utils.udp_outgauge_utility import get_telemetry

class GearShifterModule(ModuleBase):
    """
    Renders a 5-pixel-tall layout:

        1   3   5
        |---|---|
        2   4   6   R

    with vertical bars spanning y=0..4, crossbar at y=2, 
    and gear endpoints at:

      G1=(1,0), G3=(3,0), G5=(5,0)
      G2=(1,4), G4=(3,4), G6=(5,4), R=(7,4)

    OutGauge gear is remapped so:
      OutGauge=1 => (neutral or none),
      OutGauge=2 => G1,
      OutGauge=3 => G2,
      OutGauge=4 => G3,
      OutGauge=5 => G4,
      OutGauge=6 => G5,
      OutGauge=7 => G6 (if you have a 7th gear),
      OutGauge=-1 => R
    """

    def __init__(self, height=5):
        """
        :param height: Must be 5 to maintain the described arrangement.
        """
        super().__init__(height)

        # Define the final "display gear" -> (x,y) positions in a 9×5 area
        # Note: We label them G1..G6, R. The letters themselves are just conceptual.
        self.display_positions = {
            "G1": (1, 0),
            "G2": (1, 4),
            "G3": (3, 0),
            "G4": (3, 4),
            "G5": (5, 0),
            "G6": (5, 4),
            "R":  (7, 4)
        }

        # OutGauge gear -> displayed endpoint key
        # e.g., gear=2 => "G1", gear=3 => "G2", etc.
        self.gear_map = {
            2:  "G1",  # 1st gear
            3:  "G2",  # 2nd gear
            4:  "G3",  # 3rd gear
            5:  "G4",  # 4th gear
            6:  "G5",  # 5th gear
            7:  "G6",  # 6th gear (if your car has it)
            -1: "R"    # Reverse
            # gear=1 => neutral => no bright endpoint
            # if you want neutral to highlight "G1", just re-map it here
        }

    def render(self, width):
        """
        Draw the vertical lines from y=0..4 at x=1,3,5 and 
        one horizontal crossbar at y=2 from x=1..5, plus the 
        gear endpoints. Light up the current gear endpoint in bright white.
        """
        image = Image.new('L', (width, self.height), 0)

        # Retrieve the current OutGauge gear
        telemetry = get_telemetry()
        outgauge_gear = telemetry.get("gear", 0)  # 1..n or -1 for Reverse

        # Colors
        dim_color = 10
        bright_color = 255

        # 1) Draw the vertical bars at x=1,3,5 for y=0..4
        for x in [1, 3, 5]:
            for y in range(0, 5):  # covers y=0..4
                if 0 <= x < width and 0 <= y < self.height:
                    image.putpixel((x, y), dim_color)

        # 2) Draw the horizontal line at y=2 from x=1..5
        for x in range(1, 6):
            if 0 <= x < width and 0 <= 2 < self.height:
                image.putpixel((x, 2), dim_color)

        # 3) Determine which endpoint to highlight
        #    via the gear_map
        if outgauge_gear in self.gear_map:
            endpoint_key = self.gear_map[outgauge_gear]  # e.g. "G1" 
            if endpoint_key in self.display_positions:
                gx, gy = self.display_positions[endpoint_key]
                if 0 <= gx < width and 0 <= gy < self.height:
                    image.putpixel((gx, gy), bright_color)
        else:
            # If gear=1 => neutral => do nothing (or pick a fallback)
            pass

        return image
