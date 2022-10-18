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
        '--audio_files',
        nargs='+',
        required=True,
        help='The audio file to process')
    parser.add_argument(
        '--output_files',
        nargs='+',
        required=True,
        help='The file to save pitch')

    # Optionally save periodicity
    parser.add_argument(
        '--output_periodicity_files',
        nargs='+',
        help='The files to save periodicity')

    # Optionally customize configs
    parser.add_argument(
        '--config',
        default='default',
        help='Customized configurations'
    )

    return parser.parse_args()


def load_config(config: str):
    """Load configurations"""
    config_file = config if config.endswith('.json') else f'{config}.json'
    path = os.path.join(os.path.dirname(__file__), 'configs', config_file)

    with open(path, 'r', encoding='utf-8') as cfg:
        config = json.load(cfg)

    return config


def make_parent_directory(file):
    """Create parent directory for file if it does not already exist"""
    parent = os.path.dirname(os.path.abspath(file))
    os.makedirs(parent, exist_ok=True)


def main():
    # Parse command-line arguments
    args = parse_args()

    # Ensure output directory exist
    [make_parent_directory(file) for file in args.output_files]
    if args.output_periodicity_files is not None:
        [make_parent_directory(file) for file in args.output_periodicity_files]
    
    # Load configurations
    config = load_config(args.config)

    # Get decoder
    if config['decoder'] == 'argmax':
        decoder = onnxcrepe.decode.argmax
    elif config['decoder'] == 'weighted_argmax':
        decoder = onnxcrepe.decode.weighted_argmax
    elif config['decoder'] == 'viterbi':
        decoder = onnxcrepe.decode.viterbi
    else:
        raise NotImplementedError('Decoder must be \'argmax\', \'weighted_argmax\' or \'viterbi\'.')

    # Create inference session
    providers = [(provider['name'], provider['options']) for provider in config['providers']]
    available_providers_selected = []
    for provider in providers:
        if provider[0] in ort.get_available_providers():
            available_providers_selected.append(provider)
    if not available_providers_selected:
        raise NotImplementedError('None of the selected execution providers is available on this machine.')

    options = ort.SessionOptions()
    if available_providers_selected[0][0] == 'DmlExecutionProvider':
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    session = CrepeInferenceSession(model=config['model'], sess_options=options, providers=available_providers_selected)

    # Infer pitch and save to disk
    onnxcrepe.predict_from_files_to_files(session,
                                          args.audio_files,
                                          args.output_files,
                                          args.output_periodicity_files,
                                          config['precision'],
                                          config['fmin'],
                                          config['fmax'] if config['fmax'] is not None else onnxcrepe.MAX_FMAX,
                                          decoder,
                                          config['batch_size'],
                                          config['pad'])


# Run module entry point
if __name__ == '__main__':
    main()
