import argparse
import json
import os

import onnxruntime as ort

import onnxcrepe
from onnxcrepe.session import CrepeInferenceSession


###############################################################################
# Entry point
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument(
        'audio_files',
        nargs='+',
        help='The audio file to process')

    # Optional arguments
    parser.add_argument(
        '--output_directory',
        required=False,
        help='The directory to save output')  # Defaults to directory of each audio file
    parser.add_argument(
        '--save_periodicity',
        required=False,
        action='store_true',
        help='Whether save periodicity')
    parser.add_argument(
        '--format',
        required=False,
        default='csv',
        help='Saving format of the result (csv or npy)')  # Combined .csv or separated .npy
    parser.add_argument(
        '--config',
        required=False,
        default='default',
        help='Customized configurations')

    return parser.parse_args()


def load_config(config: str):
    """Load configurations"""
    config_file = config if config.endswith('.json') else f'{config}.json'
    path = os.path.join(os.path.dirname(__file__), 'configs', config_file)

    with open(path, 'r', encoding='utf-8') as cfg:
        config = json.load(cfg)

    return config


def main():
    # Parse command-line arguments
    args = parse_args()

    # Check saving format
    if args.format != 'csv' and args.format != 'npy':
        raise NotImplementedError('Saving format must be \'csv\' or \'npy\'.')

    # Ensure output directory exist
    if args.output_directory is not None:
        os.makedirs(args.output_directory, exist_ok=True)

    # Load configurations
    config = load_config(args.config)

    # Check model capacity
    if config['model'] not in ['full', 'large', 'medium', 'small', 'tiny']:
        raise NotImplementedError(
            'Model capacity must be \'full\', \'large\', \'medium\', \'small\' or \'tiny\'.')

    # Get decoder
    if config['decoder'] == 'argmax':
        decoder = onnxcrepe.decode.argmax
    elif config['decoder'] == 'weighted_argmax':
        decoder = onnxcrepe.decode.weighted_argmax
    elif config['decoder'] == 'viterbi':
        decoder = onnxcrepe.decode.viterbi
    else:
        raise NotImplementedError('Decoder must be \'argmax\', \'weighted_argmax\' or \'viterbi\'.')

    # Filter and parse providers
    available_providers_selected = []
    for provider in config['providers']:
        if provider['name'] in ort.get_available_providers():
            available_providers_selected.append(provider)
        else:
            print(f'{provider["name"]} is not available on this machine. Skipping.')

    if not available_providers_selected:
        raise NotImplementedError('None of the selected execution providers is available on this machine.')
    providers = [(provider['name'], provider['options']) for provider in available_providers_selected]

    # Create session options
    # DirectML does not support memory pattern optimizations or parallel execution in onnxruntime. See
    # https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html#configuration-options
    options = ort.SessionOptions()
    if available_providers_selected[0]['name'] == 'DmlExecutionProvider':
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    # Create inference session
    session = CrepeInferenceSession(model=config['model'], sess_options=options, providers=providers)

    # Infer pitch and save to disk
    onnxcrepe.predict_from_files_to_files(session,
                                          args.audio_files,
                                          args.output_directory,
                                          args.save_periodicity,
                                          args.format,
                                          config['precision'],
                                          config['fmin'],
                                          config['fmax'] if config['fmax'] is not None else onnxcrepe.MAX_FMAX,
                                          decoder,
                                          config['batch_size'],
                                          config['pad'])


# Run module entry point
if __name__ == '__main__':
    main()
