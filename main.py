# coding = 'utf-8'
import numpy as np
import pandas as pd
import time
import tm

def main():
    x = np.random.randint(100, size=(100000,1))
    y = np.random.randint(2, size=(100000,1))
    data = pd.DataFrame(np.concatenate([x,y], axis=1), columns = ['x','y'])

    start = time.time()
    result_1 = tm.target_mean_v3(data, 'x', 'y')
    end = time.time()
    print(end -start)

if __name__ == '__main__':
    main()
