
# import from numpy


```python
import numpy as np
import torch
```

```python
a = np.array([2,3.3])
torch.from_numpy(a)
```


    tensor([2.0000, 3.3000], dtype=torch.float64)


```python
b = np.ones([2,3])
torch.from_numpy(b)
```


    tensor([[1., 1., 1.],
            [1., 1., 1.]], dtype=torch.float64)

# import from list

```python
torch.tensor([2,3.2])
```


    tensor([2.0000, 3.2000])


```python
torch.FloatTensor(2,3,4)
```


    tensor([[[8.9082e-39, 5.9694e-39, 8.9082e-39, 1.0194e-38],
             [9.1837e-39, 4.6837e-39, 9.9184e-39, 9.0000e-39],
             [1.0561e-38, 1.0653e-38, 4.1327e-39, 8.9082e-39]],
    
            [[9.8265e-39, 9.4592e-39, 1.0561e-38, 6.7041e-39],
             [1.1112e-38, 9.5511e-39, 1.0102e-38, 9.0918e-39],
             [1.0469e-38, 8.4490e-39, 1.0102e-38, 9.2755e-39]]])


```python
torch.FloatTensor([2,3])
```


    tensor([2., 3.])


```python
type(torch.FloatTensor([2,3]))
```


    torch.Tensor




```python
type(torch.FloatTensor(2,3))
```


    torch.Tensor


```python
torch.tensor([[2,3.2],[1,22.3]])
```


    tensor([[ 2.0000,  3.2000],
            [ 1.0000, 22.3000]])


```python
# uninitialized
```


```python
torch.FloatTensor(3,1)
```

    tensor([[0.0000],
            [2.9375],
            [   nan]])


```python
# set default type
```


```python
torch.tensor([1.2,3]).type()
```


    'torch.FloatTensor'


```python
torch.IntTensor([1.2,3]).type()
```


    'torch.IntTensor'


```python
# 随机初始化
```

```python
torch.rand(3,3)
```




    tensor([[0.9773, 0.1516, 0.2925],
            [0.5249, 0.1829, 0.5700],
            [0.4827, 0.3933, 0.6341]])




```python
a = torch.rand(3,3)
torch.rand_like(a)
```




    tensor([[0.3773, 0.3175, 0.8735],
            [0.9210, 0.2805, 0.7514],
            [0.3408, 0.3948, 0.9681]])




```python
torch.randint(1, 10, [3,3])
```




    tensor([[3, 9, 6],
            [2, 5, 1],
            [5, 3, 5]])




```python
# randn
torch.randn(3, 3)
```




    tensor([[ 0.3305,  0.4621, -1.4146],
            [-0.1905, -0.5162, -1.4178],
            [-1.1602,  1.0984,  1.0032]])




```python
torch.normal(mean=torch.full([10],0),std = torch.arange(1,0,-0.1))
```




    tensor([ 0.2889,  0.9067, -0.4615,  0.2284,  1.1094,  0.2070,  0.0618, -0.4863,
             0.2777, -0.0510])




```python
# full
torch.full([2,3],7)
```




    tensor([[7., 7., 7.],
            [7., 7., 7.]])




```python
torch.full([],7)
```




    tensor(7.)




```python
torch.full([1],7)
```




    tensor([7.])




```python
torch.full([3],7)
```




    tensor([7., 7., 7.])




```python
torch.linspace(0,10,steps = 11)
```




    tensor([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.])




```python
torch.logspace(0,1,steps=11)
```




    tensor([ 1.0000,  1.2589,  1.5849,  1.9953,  2.5119,  3.1623,  3.9811,  5.0119,
             6.3096,  7.9433, 10.0000])




```python
torch.ones(3,3)
```




    tensor([[1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.]])




```python
torch.zeros(3,3)
```




    tensor([[0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.]])




```python
a = torch.eye(3,3)
```


```python
torch.ones_like(a)
```




    tensor([[1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.]])




```python
idx=torch.randperm(2)
idx
```




    tensor([0, 1])




```python
a = torch.eye(2,2)
a[idx]
```




    tensor([[1., 0.],
            [0., 1.]])




```python

```


```python

```


```python

```
