import os
import torch
import shutil
import pyfiglet
from functools import partial


torch_load = partial(torch.load, map_location='cpu', weights_only=True)


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def print_message(message: str):
    try:
        terminal_width = shutil.get_terminal_size().columns
    except:
        terminal_width = 50
    print('\n' + '-' * terminal_width)
    print(f'\n{message}\n')
    print('-' * terminal_width + '\n')


def print_title(title: str):
    print(pyfiglet.figlet_format(title, font='3d-ascii'))


def print_done():
    print(pyfiglet.figlet_format('== Done ==', font='js_stick_letters'))


def expand_dms_ids_all(dms_ids, mode: str = None):
    """
    Expand 'all' to actual DMS IDs from benchmarks.proteingym.dms_ids.
    """
    if any(str(x).lower() == 'all' for x in dms_ids):
        if mode == 'indels':
            from benchmarks.proteingym.dms_ids import ALL_INDEL_DMS_IDS
            dms_ids = list(ALL_INDEL_DMS_IDS)
        else:
            from benchmarks.proteingym.dms_ids import ALL_SUBSTITUTION_DMS_IDS
            dms_ids = list(ALL_SUBSTITUTION_DMS_IDS)
    return dms_ids


def maybe_compile(model: torch.nn.Module):
    if os.name == 'posix':
        try:
            torch.compile(model, dynamic=True)
            print_message("Model compiled")
        except:
            print_message("Not linux system, will not compile model")
    return model


if __name__ == '__main__':
    folders_to_clean = ['logs', 'results', 'plots', 'embeddings', 'weights']
    
    for folder in folders_to_clean:
        if os.path.exists(folder):
            files = os.listdir(folder)
            if files:
                response = input(f"Do you want to delete all files in '{folder}' folder? ({len(files)} files) [y/N]: ")
                if response.lower() == 'y':
                    for file in files:
                        file_path = os.path.join(folder, file)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                    print(f"All files in '{folder}' have been deleted.")
                else:
                    print(f"Skipped cleaning '{folder}' folder.")
            else:
                print(f"'{folder}' folder is already empty.")
        else:
            print(f"'{folder}' folder does not exist.")
