import math
import multiprocessing
import re
import subprocess as sp
import time
import ffmpeg
import numpy as np
import torch

from .bg import DEVICE, Net, iter_frames, remove_many


def worker(worker_nodes,
           worker_index,
           result_dict,
           model_name,
           gpu_batchsize,
           total_frames,
           frames_dict):
    print(F"WORKER {worker_index} ONLINE")

    output_index = worker_index + 1
    base_index = worker_index * gpu_batchsize
    net = Net(model_name)
    script_net = None
    for fi in (list(range(base_index + i * worker_nodes * gpu_batchsize,
                          min(base_index + i * worker_nodes * gpu_batchsize + gpu_batchsize, total_frames)))
               for i in range(math.ceil(total_frames / worker_nodes / gpu_batchsize))):
        if not fi:
            break

        # are we processing frames faster than the frame ripper is saving them?
        last = fi[-1]
        while last not in frames_dict:
            time.sleep(0.1)

        input_frames = [frames_dict[index] for index in fi]
        if script_net is None:
            script_net = torch.jit.trace(net,
                                         torch.as_tensor(np.stack(input_frames), dtype=torch.float32, device=DEVICE))


        result_dict[output_index] = remove_many(input_frames, script_net)

        # clean up the frame buffer
        for fdex in fi:
            del frames_dict[fdex]
        output_index += worker_nodes


def capture_frames(file_path, frames_dict, prefetched_samples, total_frames):
    print(F"WORKER FRAMERIPPER ONLINE")
    for idx, frame in enumerate(iter_frames(file_path)):
        frames_dict[idx] = frame
        while len(frames_dict) > prefetched_samples:
            time.sleep(0.1)
        if idx > total_frames:
            break


def parallel_greenscreen(file_path,
                         worker_nodes,
                         gpu_batchsize,
                         model_name,
                         frame_limit=-1,
                         prefetched_batches=4,
                         framerate=-1):
    manager = multiprocessing.Manager()

    results_dict = manager.dict()
    frames_dict = manager.dict()

    print(file_path)

    info = ffmpeg.probe(file_path)
    total_frames = int(info["streams"][0]["nb_frames"])

    if frame_limit != -1:
        total_frames = min(frame_limit, total_frames)

    fr = info["streams"][0]["r_frame_rate"]

    if framerate == -1:
        print(F"FRAME RATE DETECTED: {fr} (if this looks wrong, override the frame rate)")
        framerate = math.ceil(eval(fr))

    print(F"FRAME RATE: {framerate} TOTAL FRAMES: {total_frames}")

    p = multiprocessing.Process(target=capture_frames,
                                args=(file_path, frames_dict, gpu_batchsize * prefetched_batches, total_frames))
    p.start()

    # note I am deliberatley not using pool
    # we can't trust it to run all the threads concurrently (or at all)
    workers = [multiprocessing.Process(target=worker,
                                       args=(worker_nodes, wn, results_dict, model_name, gpu_batchsize, total_frames,
                                             frames_dict))
               for wn in range(worker_nodes)]
    for w in workers:
        w.start()

    command = None
    proc = None
    frame_counter = 0
    for i in range(math.ceil(total_frames / worker_nodes)):
        for wx in range(worker_nodes):

            hash_index = i * worker_nodes + 1 + wx

            while hash_index not in results_dict:
                time.sleep(0.1)

            frames = results_dict[hash_index]
            # dont block access to it anymore
            del results_dict[hash_index]

            for frame in frames:
                if command is None:
                    command = ['ffmpeg',
                               '-y',
                               '-f', 'rawvideo',
                               '-vcodec', 'rawvideo',
                               '-s', F"{frame.shape[1]}x320",
                               '-pix_fmt', 'gray',
                               '-r', F"{framerate}",
                               '-i', '-',
                               '-an',
                               '-vcodec', 'mpeg4',
                               '-b:v', '2000k',
                               re.sub(r"\.(mp4|mov|avi)", r".matte.\1", file_path, flags=re.I)]

                    proc = sp.Popen(command, stdin=sp.PIPE)

                proc.stdin.write(frame.tostring())
                frame_counter = frame_counter + 1

                if frame_counter >= total_frames:
                    p.join()
                    for w in workers:
                        w.join()
                    proc.stdin.close()
                    proc.wait()
                    print(F"FINISHED ALL FRAMES ({total_frames})!")
                    return

    p.join()
    for w in workers:
        w.join()
    proc.stdin.close()
    proc.wait()
