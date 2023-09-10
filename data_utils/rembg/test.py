from .multiprocessing import parallel_greenscreen

if __name__ == "__main__":

    parallel_greenscreen("C:\\Users\\tim\\Videos\\test\\2021-01-31 14-05-36.mp4", 
        3, 
        1, 
        "u2net_human_seg")
