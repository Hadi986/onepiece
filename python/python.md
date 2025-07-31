

## 高级特性

### 切片

### 迭代

### 列表生成式
```python
[x * x for x in range(1, 11)]
[1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
```

### 生成器（generator）
```python
>>> g = (x * x for x in range(10))
>>> g
<generator object <genexpr> at 0x1022ef630>
```
相较于列表生成式，生成器优势在于：内存效率，惰性计算避免一次性加载所有数据

### 迭代器


## 函数式编程(Functional Programming)

### 高阶函数
- map/reduce
map()函数接收两个参数，一个是函数，一个是Iterable，map将传入的函数依次作用到序列的每个元素，并把结果作为新的Iterator返回。
```python
>>> def f(x):
...     return x * x
...
>>> r = map(f, [1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> list(r)
[1, 4, 9, 16, 25, 36, 49, 64, 81]
```
reduce把一个函数作用在一个序列[x1, x2, x3, ...]上，这个函数必须接收两个参数，reduce把结果继续和序列的下一个元素做累积计算，其效果就是：
```python
>>> from functools import reduce
>>> def add(x, y):
...     return x + y
...
>>> reduce(add, [1, 3, 5, 7, 9])
25
```

- filter
filter()也接收一个函数和一个序列。和map()不同的是，filter()把传入的函数依次作用于每个元素，然后根据返回值是True还是False决定保留还是丢弃该元素。
```python
def is_odd(n):
    return n % 2 == 1

list(filter(is_odd, [1, 2, 4, 5, 6, 9, 10, 15]))
# 结果: [1, 5, 9, 15]
```

- sorted

### 闭包（Closure）
闭包是 Python 中的嵌套函数，其中内层函数引用了外层函数的变量，并且外层函数返回这个内层函数。闭包的主要特点是，当外层函数执行完毕后，内层函数仍然可以访问和修改外层函数的局部变量。
```python
def lazy_sum(*args):
    def sum():
        ax = 0
        for n in args:
            ax = ax + n
        return ax
    return sum
>>> f = lazy_sum(1, 3, 5, 7, 9)
>>> f //求和函数
<function lazy_sum.<locals>.sum at 0x101c6ed90>
>>> f() //求和的结果
25
闭包的使用场景包括但不限于以下几种：
- 数据封装：闭包可以用来封装数据，隐藏内部实现细节，只暴露必要的接口。
- 函数工厂：可以生成具有不同行为的函数。
- 装饰器：闭包是 Python 装饰器的基础，装饰器可以用来增强或修改函数的行为。
- 延迟计算：可以延迟某些计算，直到需要时再执行。
闭包的基本语法
- 闭包通常由以下几个部分组成：
- 外层函数：定义了局部变量和内层函数。
- 内层函数：引用了外层函数的局部变量。
- 返回值：外层函数返回内层函数。
```

### 匿名函数（lambda表达式）
```python
>>> list(map(lambda x: x * x, [1, 2, 3, 4, 5, 6, 7, 8, 9]))
[1, 4, 9, 16, 25, 36, 49, 64, 81]
==========
def f(x):
    return x * x
```

### 装饰器
允许用户在不修改函数定义的情况下，增强或修改函数的功能。装饰器本质上是一个接受函数作为参数并返回新函数的函数。装饰器通常用于日志记录、访问控制、性能计时等场景。
装饰器通常定义为一个包装函数，该包装函数接受一个函数作为参数，并返回一个新的函数。新的函数通常会在调用原始函数之前或之后执行一些额外的操作。

Python 中的装饰器是一种非常强大的工具，它允许用户在不修改函数定义的情况下，增强或修改函数的功能。装饰器本质上是一个接受函数作为参数并返回新函数的函数。装饰器通常用于日志记录、访问控制、性能计时等场景。


#### 装饰器的使用示例

以下是一个简单的装饰器示例，用于计算函数的执行时间：

```python
import time

def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time:.4f} seconds to execute.")
        return result
    return wrapper

@timer_decorator
def example_function(n):
    sum = 0
    for i in range(n):
        sum += i
    return sum

print(example_function(1000000))
```

在这个例子中，我们定义了一个装饰器 `timer_decorator`，它接受一个函数 `func` 作为参数。装饰器内部定义了一个包装函数 `wrapper`，它在调用原始函数之前记录开始时间，在调用之后记录结束时间，并计算函数的执行时间。`wrapper` 函数返回原始函数的返回值。
`@timer_decorator def example_function(n):`等价于`example_function = timer_decorator(example_function)`
由于`timer_decorator()`是一个decorator，返回一个函数，所以，原来的`example_function()`函数仍然存在，只是现在同名的`example_function`变量指向了新的函数，于是调用`example_function()`将执行新函数，即在`timer_decorator()`函数中返回的`wrapper()`函数。
`wrapper()`函数的参数定义是`(*args, **kwargs)`，因此，`wrapper()`函数可以接受任意参数的调用。在`wrapper()`函数内，首先打印日志，再紧接着调用原始函数。
我们通过 `@timer_decorator` 语法将装饰器应用到 `example_function` 函数上。当 `example_function` 被调用时，实际上执行的是装饰器返回的 `wrapper` 函数，从而实现了对原始函数的功能增强。

#### 带参数的装饰器

装饰器也可以接受参数，这使得装饰器更加灵活。以下是一个带参数的装饰器示例：

```python
def repeat_decorator(repeat_count):
    def decorator(func):
        def wrapper(*args, **kwargs):
            results = []
            for _ in range(repeat_count):
                results.append(func(*args, **kwargs))
            return results
        return wrapper
    return decorator

@repeat_decorator(repeat_count=3)
def greet(name):
    return f"Hello, {name}!"

print(greet("Alice"))
```

在这个例子中，我们定义了一个带参数的装饰器 `repeat_decorator`，它接受一个参数 `repeat_count`。装饰器返回一个内部函数 `decorator`，该函数又返回一个包装函数 `wrapper`。`wrapper` 函数多次调用原始函数，并收集结果到一个列表中返回。

通过 `@repeat_decorator(repeat_count=3)` 语法，我们将装饰器应用到 `greet` 函数上。当 `greet` 函数被调用时，它会重复执行 3 次，并返回一个包含三次执行结果的列表。

#### 装饰器的注意事项

- **保持函数签名一致**：装饰器返回的函数应该与原始函数具有相同的参数列表和返回值类型。
- **使用 `functools.wraps`**：为了保留原始函数的元信息（如函数名、文档字符串等），可以使用 `functools.wraps` 装饰器。

```python
import functools

def timer_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time:.4f} seconds to execute.")
        return result
    return wrapper
```

`functools.wraps` 是一个方便的装饰器，它帮助我们保留原始函数的元信息。

装饰器是 Python 中一个非常有用的特性，它允许我们以非侵入的方式增强函数的功能。通过合理使用装饰器，可以使代码更加简洁、模块化和可维护。

### 偏函数
偏函数（Partial Function）是指通过固定原函数的部分参数，创建一个新的函数。这个新函数保留了原函数的核心功能，但减少了需要传递的参数数量。

核心概念
- 参数绑定：将函数的部分参数预先绑定特定值
- 函数特化：创建更具体、更易用的函数版本
- 接口简化：减少调用时需传递的参数数量

```python
# 原始函数
def power(base, exponent):
    return base ** exponent

# 创建偏函数：固定 exponent 为 2
square = partial(power, exponent=2)

# 创建另一个偏函数：固定 base 为 10
power_of_10 = partial(power, base=10)

print(square(5))      # 25 (5^2)
print(square(7))      # 49 (7^2)
print(power_of_10(3)) # 1000 (10^3)
print(power_of_10(5)) # 100000 (10^5)
```