import functools

def my_decorator(func):
    def wrapper(*args, **kwargs):
        '''decorator'''
        print('Calling decorated function...')
        return func(*args, **kwargs)

    return wrapper


@my_decorator
def example():
    """Docstring"""
    print('Called example function')


print('No functools wrap name: {}'.format(example.__name__))
print('No functools wrap docstring: {}'.format(example.__doc__))

def my_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        '''decorator'''
        print('Calling decorated function...')
        return func(*args, **kwargs)

    return wrapper


@my_decorator
def example():
    """Docstring"""
    print('Called example function')

print('name: {}'.format(example.__name__))
print('docstring: {}'.format(example.__doc__))