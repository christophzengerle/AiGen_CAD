import os

from OCC.Extend.DataExchange import write_step_file


def step_file_exists(path):
    return os.path.isfile(path)


def create_step_file(out_shape, path):
    try:
        if not step_file_exists(path):
            write_step_file(out_shape, path)
            print("{} created.".format(path.split("/")[-1]))

        else:
            print(f".STEP-File {path.split('/')[-1]} already exists.")

    except Exception as e:
        raise Exception(
            f"Creation of .STEP-File for {path.split('/')[-1]} failed.\n"
            + str(e.with_traceback)
        )
