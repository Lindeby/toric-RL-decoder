from src.nn.torch.ResNet import ResNet18
import numpy as np
import torch, time

if __name__ == "__main__":

    system_size = 9
    model = ResNet18()
    batch_size = 100

    f1 = 0

    with torch.no_grad():

        for b in range(1, batch_size+1):
            batch = np.ones((b, 2, system_size, system_size))
            t_batch = torch.from_numpy(batch).type(torch.Tensor)
            start = time.time()
            model(t_batch)
            end = time.time()

            if b == 1:
                f1 = end-start
            fb = end-start
            print("Batch size {} took {}. Gained {} percent of the time.".format(b, fb, fb/(f1*b)))