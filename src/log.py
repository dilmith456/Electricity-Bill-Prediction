import logging
import os
from datetime import datetime

#Define the Logfile Name using current Date and Time
log_file_name = f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log"

#Create the path to the logfile
log_file_path = os.path.join(os.getcwd(), 'logs', log_file_name)

#Create the logs directory
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

#Configuration
logging.basicConfig(
    filename = log_file_path,
    #Set the logging level to INFO - Only the info messages will be raised
    level=logging.INFO,
    format= "[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s"
)
 
# #Test the logger
# if __name__ == "__main__":
#     logging.info("Logging in to the system")