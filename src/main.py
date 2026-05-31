import argparse

from .generation.dataset_generator import main as gen
from .simulation.sim_pipeline import main as sim
from .viewer.view import main as view


def args_parse():
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers()

    # Microscope arguments
    parser_micr = argparse.ArgumentParser(add_help=False)
    parser_micr.add_argument("--pattern_case", type=str, default="multipoint")
    parser_micr.add_argument(
        "--device",
        type=str,
        default="cuda",
    )
    parser_micr.add_argument("--resolution", type=tuple, default=(512, 512))
    # parser_micr.add_argument("--cam_height", type=int, default=512)
    # parser_micr.add_argument("--cam_width", type=int, default=512)
    parser_micr.add_argument("--cam_pix", type=float, default=6.5)
    parser_micr.add_argument("--binn_simu", type=int, default=2)
    parser_micr.add_argument("--single_plane", action="store_false")
    parser_micr.add_argument("--axial_fov", type=int, default=30)
    parser_micr.add_argument("--periodicity", type=int, default=1)
    parser_micr.add_argument("--wavelength_ex", type=int, default=488)
    parser_micr.add_argument("--wavelength_em", type=int, default=520)
    parser_micr.add_argument("--phase_optim", type=int, default=1)
    parser_micr.add_argument("--aod_propag", type=int, default=1)
    parser_micr.add_argument("--numerical_aperture", type=float, default=1.2)
    parser_micr.add_argument("--magnification", type=int, default=60)
    parser_micr.add_argument("--focal", type=int, default=200)
    parser_micr.add_argument("--pix_size_simu_axi", type=int, default=200)
    parser_micr.add_argument("--upsampling_factor", type=float, default=1.05)
    parser_micr.add_argument("--inpainting_mask", type=float, default=0.8)
    parser_micr.add_argument("--noise_level_gaussian", type=float, default=0.1)
    parser_micr.add_argument("--noise_level_poisson", type=float, default=0.1)

    # Noise arguments
    parser_noise = argparse.ArgumentParser(add_help=False)
    parser_noise.add_argument("--inpainting_noise", type=float, default=0.8)
    parser_noise.add_argument("--gaussian_noise", type=float, default=0.1)

    # Dataset arguments
    parser_data = argparse.ArgumentParser(add_help=False)
    parser_data.add_argument("--source_path", type=str, default="data/GTs/")
    parser_data.add_argument("--batch_size", type=int, default=1, help="Only change if input images have the same size. Default: 1.")

    # Simulation arguments
    parser_simulation = subparser.add_parser(
        "sim", parents=[parser_micr, parser_noise, parser_data]
    )
    parser_simulation.add_argument("--output_path", type=str, default=None)
    parser_simulation.add_argument("--visualize", action="store_true")
    parser_simulation.set_defaults(func=sim)

    # Generations arguments
    parser_generations = subparser.add_parser("gen")
    parser_generations.add_argument(
        "--config_shape", type=str, default="configs/shapes/sample_dataset.yaml"
    )
    parser_generations.add_argument(
        "--output_folder", type=str, default="data/synth_dataset/"
    )
    parser_generations.set_defaults(func=gen)

    # Viewer arguments
    parser_viewer = subparser.add_parser("view", parents=[parser_data])
    parser_viewer.set_defaults(func=view)

    args = parser.parse_args()

    return args


def main(args):
    args.func(args)


if __name__ == "__main__":
    args = args_parse()
    main(args)
