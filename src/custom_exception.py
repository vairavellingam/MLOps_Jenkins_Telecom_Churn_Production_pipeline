import sys


class CustomException(Exception):
    def __init__(self, message: str, error_detail: Exception = None):
        self.error_message = self.get_detailed_error_message(message, error_detail)
        super().__init__(self.error_message)

    @staticmethod
    def get_detailed_error_message(message: str, error_detail):
        _, _, exc_tb = sys.exc_info()
        if exc_tb:
            file_name = exc_tb.tb_frame.f_code.co_filename
            line_number = exc_tb.tb_lineno
        else:
            file_name = "N/A"
            line_number = "N/A"

        return (
            f"{message} | "
            f"Error: {error_detail} | "
            f"File: {file_name} | "
            f"Line: {line_number}"
        )

    def __str__(self):
        return self.error_message
