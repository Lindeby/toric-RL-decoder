from datetime import datetime

def logger(message_queue):

    start = datetime.now()
    with open('../logs/log_{}'.format(start.strftime("%d-%m-%Y_%H:%M:%S")), 'w') as f:
        try:
            while True:        
                if not message_queue.empty():
                    message = message_queue.get()
                    now = datetime.now()
                    f.write(now.strftime("%d-%m-%Y_%H:%M:%S") + "    {}" +  "\n".format(str(message)))
        except Exception as e:
            f.write(start.strftime("%d-%m-%Y_%H:%M:%S") + "    " + str(e) +  "\n")
