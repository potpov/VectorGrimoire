import yaml
import os
import argparse

def main(args):
    with open("configs/VSQ_sweep/configs.yaml", 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    num_configs = len(config)

    with open("configs/VSQ_sweep/Base_VSQ_config.yaml", 'r') as file:
        try:
            base_vsq_config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    # Access the parsed arguments
    save_base_dir = args.save_base_dir

    for i in range(num_configs):
        config_id = list(config.keys())[i]
        config_params = config[config_id]

        # Create a new config dictionary
        new_config = base_vsq_config

        # Update the new config dictionary with the config parameters
        new_config["model_params"]["num_codes_per_shape"] = config_params["num_codes_per_shape"]
        new_config["model_params"]["num_segments"] = config_params["num_segments"]
        new_config["model_params"]["geometric_constraint"] = config_params["geometric_constraint"]
        new_config["data_params"]["individual_max_length"] = config_params["individual_max_length"]
        new_config["data_params"]["stroke_width"] = config_params["stroke_width"]
        new_config["logging_params"]["save_dir"] = os.path.join(save_base_dir, f"VSQ_{config_id}")
        new_config["logging_params"]["name"] = f"Figr8 VSQ_{config_id}"


        # Create a new directory for the config
        save_dir = "configs/VSQ_sweep"

        # Create the directory
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save the new config dictionary as a yaml file
        with open(os.path.join(save_dir, f"VSQ_{config_id}.yaml"), 'w') as file:
            try:
                yaml.dump(new_config, file)
            except yaml.YAMLError as exc:
                print(exc)

if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser()

    # Add command line arguments
    parser.add_argument("--save_base_dir", type=str, default="/scratch2/moritz_logs/VSQ", help="Save base directory argument")

    # Parse the command line arguments
    args = parser.parse_args()
    assert os.path.basename(os.getcwd()) == "thesis", "The current working directory must be the root of the thesis project for this to work"
    print("Make sure that batch_size and logging params etc are the desired values in VSQ_base.yaml before running this script as we're copying these values.")
    print(f"INFO: top level dir of checkpoint saving is set to {args.save_base_dir}")
    input("Press Enter to continue...")
    main(args)
