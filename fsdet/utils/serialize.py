# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import cloudpickle

# pickle：每种编程语言都有各自的数据类型, 将属于自己语言的数据类型或对象转换为可通过网络传输或可以存储到本地磁盘的数据格式
#        （如：XML、JSON或特定格式的字节串）的过程称为序列化(seralization)；反之则称为反序列化。
# cloudpickle:拓展了pickle的功能，让一些其他的数据类型也能被序列化。
class PicklableWrapper(object):
    """
    Wrap an object to make it more picklable, note that it uses
    heavy weight serialization libraries that are slower than pickle.
    It's best to use it only on closures (which are usually not picklable).

    This is a simplified version of
    https://github.com/joblib/joblib/blob/master/joblib/externals/loky/cloudpickle_wrapper.py
    """

    def __init__(self, obj):
        self._obj = obj

    def __reduce__(self):
        s = cloudpickle.dumps(self._obj)
        return cloudpickle.loads, (s,)

    def __call__(self, *args, **kwargs):
        return self._obj(*args, **kwargs)

    def __getattr__(self, attr):
        # Ensure that the wrapped object can be used seamlessly as the previous object.
        if attr not in ["_obj"]:
            return getattr(self._obj, attr)
        return getattr(self, attr)
