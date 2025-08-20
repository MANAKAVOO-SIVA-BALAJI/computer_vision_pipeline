import schedule
import time
from datetime import datetime, timedelta
from main import start_pipeline, stop_pipeline
import sys

def pipeline_start():
    print("Pipeline started on", datetime.now())
    start_pipeline()

def pipeline_stop():
    print("Pipeline stopped on", datetime.now())
    stop_pipeline()
    sys.exit(0)

now = datetime.now()
start_time = (now + timedelta(minutes=1)).strftime("%H:%M")  
end_time = (now + timedelta(minutes=4)).strftime("%H:%M")   

print(f"Scheduling pipeline start at {start_time} and stop at {end_time}")

schedule.every().day.at(start_time).do(pipeline_start)
schedule.every().day.at(end_time).do(pipeline_stop)

print("Pipeline scheduled tasks running...")

while True:
    schedule.run_pending()
    # time.sleep(1)
