import os
import mimetypes

def get_file_type(input_file):
    if input_file.upper() == "CAM":
        return "CAM"
    abs_path = os.path.abspath(input_file)
    mime_type = mimetypes.guess_type(abs_path)
    if "image" in mime_type[0]:
        return "IMAGE"
    elif "video" in mime_type[0]:
        return "VIDEO"
    else:
        return False