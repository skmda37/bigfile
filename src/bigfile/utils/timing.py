import time


def time_spent(tic1, tag='', count=1):
    toc1 = time.process_time()
    t = (toc1 - tic1)*100./count
    print(f"time spend on {tag} method = {t:.2f}ms")
    return t