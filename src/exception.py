import sys
from src.log import logging


def error_message_detail(error, error_details_object: sys):
    _, _, exc_tb = error_details_object.exc_info()

    file_name = exc_tb.tb_frame.f_code.co_filename

    formatted_error_message = (
        f"Error in Python Script: [{file_name}] "
        f"Line number [{exc_tb.tb_lineno}] "
        f"Error message [{str(error)}]"
    )

    return formatted_error_message


class FileOperationError(Exception):
    def __init__(self, error, error_details_object: sys):
        message = error_message_detail(error, error_details_object)
        super().__init__(message)
        self.formatted_error_message = message

    def __str__(self):
        return self.formatted_error_message


# # Testing block (OUTSIDE class)
# if __name__ == "__main__":
#     try:
#         a = 1 / 1
#     except Exception as e:
#         logging.info("Testing custom exception")
#         raise FileOperationError(e, sys)
