from io import StringIO
from string import ascii_uppercase
from typing import Any, Callable, Mapping, NotRequired, Required, TypeVar
from types import GenericAlias, UnionType, NoneType
import json, re

__all__ = ['Transform', 'DataParcel', 'Parcelable', 'JSONConverter', 'parse_variable']

Transform = Callable[[Any, 'Registry'], Any]
Registry = Mapping['TypeLike', Transform]
_T = TypeVar('_T')

def _snake2camel(value: str) -> str:
    s = StringIO()
    upper_next = False
    for char in value:
        if char == '_':
            upper_next = True
        elif upper_next:
            s.write(char.upper())
            upper_next = False
        else:
            s.write(char)
    return s.getvalue()

def _camel2snake(value: str) -> str:
    s = StringIO()
    for char in value:
        if char in ascii_uppercase:
            s.write('_')
        s.write(char.lower())
    return s.getvalue()

class DataParcel:
    '''
    A typed dictionary, with each key-value pair accessible via attribute get/set.

    Example
    -------
    >>> types = { 'a': int, 'optional_string': str | None }
    >>> parcel = DataParcel('MyClass', types, { 'a': 1, 'optional_string': 'hello' })
    >>> parcel.a
    1
    >>> parcel.to_dict() # converted to camelCase
    { 'a': 1, 'optionalString': 'hello' }

    '''
    def __init__(self, name: str, types: dict[str, 'TypeLike'], initial_value: Mapping[str, Any]):
        '''
        Initializes a new parcel with given types and data.

        Parameters
        ----------
        name:
            Class name of the parcel.
        types:
            Type mapping. 
            All keys must be present in `initial_value`, unless corresponding type is a union with `None`.
            In this case, a default `None` will be set.
        initial_value:
            The data of the parcel.
        
        Raises
        ------
        TypeError:
            Raised if one of the following occurs:
            - If some keys in `types` are missing from `initial_value` and corresponding type is not a union with `None`.
            - If redundant keys are found in `initial_value`.
        '''
        seems_missing = frozenset(types) - frozenset(initial_value)
        really_missing = set()
        for field_name in seems_missing:
            value_type = types[field_name]
            if isinstance(value_type, UnionType) and NoneType in value_type.__args__:
                initial_value[field_name] = None
            else:
                really_missing.add(field_name)
        if really_missing:
            names = ', '.join(map(repr, really_missing))
            raise TypeError('{}() missing {} required arguments: {}'.format(name, len(really_missing), names))
        redundant = frozenset(initial_value) - frozenset(types)
        if redundant:
            names = ', '.join(map(repr, redundant))
            raise TypeError('{}() got {} unexpected arguments: {}'.format(name, len(redundant), names))
        
        self._name = name
        self._types = types
        self._data = dict(initial_value)
    def __getattr__(self, attr: str):
        if attr in self._types:
            return self._data[attr]
        raise AttributeError('{!r} object has no attribute {!r}'.format(self._name, attr))
    def __setattr__(self, attr: str, value: Any):
        if attr.startswith('_'):
            self.__dict__[attr] = value
            return
        if attr in self._types:
            self._data[attr] = value
        else:
            raise AttributeError('{!r} object has no attribute {!r}'.format(self._name, attr))
    def __repr__(self):
        pairs = ', '.join('{} = {!r}'.format(k, v) for k, v in self._types.itmes())
        return f'{self._name}({pairs})'
    def __iter__(self):
        return iter(self._data)
    @classmethod
    def from_dict(cls: type[_T], name: str, types: dict[str, type], d: Mapping[str, Any], temporary_decoders: Registry = {}) -> _T:
        '''
        Recursively converts a JSON deserialized dictionary into data parcel.

        Parameters
        ----------
        name:
            Class name of the parcel.
        types:
            Type mapping.
        d:
            JSON deserialized dictionary.
            Typically comes from `JSONSerializer.decode_json`.
        temporary_decoders:
            Decoders that are not in `JSONConverter` registry.
        
        Returns
        -------
        parcel:
            The converted parcel.
        
        Raises
        ------
        TypeError:
            Raised if one of the following occurs:
            - If some keys in `types` are missing from `initial_value` and corresponding type is not a union with `None`.
            - If redundant keys are found in `initial_value`.
        '''
        data = {}
        for key, value in d.items():
            try:
                value_type = types[_camel2snake(key)]
            except KeyError:
                continue # let __init__ handle the error.
            data[_camel2snake(key)] = JSONConverter.decode(value, value_type, temporary_decoders)
        return DataParcel(name, types, data)
    def to_dict(self, temporary_encoders: Registry = {}) -> dict[str, Any]:
        '''
        Recursively converts this parcel to JSON serializable dictionary.

        Parameters
        ----------
        temporary_encoders:
            Encoders that are not in `JSONConverter` registry.
        
        Returns
        -------
        d:
            The converted dictionary.
        
        Raises
        ------
        Propagates converters' exceptions.
        '''
        d = {}
        for key, value_type in self._types.items():
            value = self._data[key]
            d[_snake2camel(key)] = JSONConverter.encode(value, value_type, temporary_encoders)
        return d

class Parcelable:
    '''
    Decorator for a parcelable class.

    Example
    -------
    >>> @Parcelable
    ... class A:
    ...     a: int
    ...     b: bool
    ...     c: str | None
    >>> # positional / keyword / optional (union with None)
    >>> a = A(1, b=True)
    >>> a.to_dict()
    { 'a': 1, 'b': True, 'c': None }
    >>> 
    '''
    @staticmethod
    def mark_dataclass(klass: type, register_subclasses: bool = False, **properties: 'TypeLike'):
        '''
        Marks a dataclass as parcelable.
        This registers a encoder & decoder pair in `JSONConverter`.

        Parameters
        ----------
        klass:
            The marked type.
        register_subclasses:
            See `JSONConverter.register`.
        properties:
            Additional properties that will be encoded.
        '''
        def encoder(value, encoders):
            d = {}
            for name in klass.__annotations__:
                d[name] = getattr(value, name)
            for name in properties:
                d[name] = getattr(value, name)
            types = klass.__annotations__.copy()
            types.update(properties)
            return DataParcel(klass.__name__, types, d).to_dict(encoders)
        def decoder(value, decoders):
            p = DataParcel.from_dict(klass.__name__, klass.__annotations__, value, decoders)
            return klass(**p._data)
        JSONConverter.register(klass, encoder, decoder, register_subclasses)
    @staticmethod
    def mark_typeddict(klass: type, register_subclasses: bool = False):
        '''
        Marks a TypedDict as parcelable.
        This registers a encoder & decoder pair in `JSONConverter`.

        Parameters
        ----------
        klass:
            The marked type.
        register_subclasses:
            See `JSONConverter.register`.
        '''
        def encoder(value, encoders):
            d = {}
            types = {}
            for name, value_type in klass.__annotations__.items():
                is_required = klass.__total__
                if isinstance(value_type, GenericAlias):
                    if value_type.__origin__ == NotRequired:
                        is_required = False
                        value_type = value_type.__args__[0]
                    elif value_type.__origin__ == Required:
                        is_required = True
                        value_type = value_type.__args__[0]
                if name in value:
                    types[name] = value_type
                    d[name] = value[name]
                elif is_required:
                    # required but missing, let __init__ handle errors
                    types[name] = value_type
            return DataParcel(klass.__name__, klass.__annotations__, value).to_dict(encoders)
        def decoder(value, decoders):
            d = {}
            types = {}
            for name, value_type in klass.__annotations__.items():
                is_required = klass.__total__
                if isinstance(value_type, GenericAlias):
                    if value_type.__origin__ == NotRequired:
                        is_required = False
                        value_type = value_type.__args__[0]
                    elif value_type.__origin__ == Required:
                        is_required = True
                        value_type = value_type.__args__[0]
                if _snake2camel(name) in value:
                    types[name] = value_type
                    d[name] = value[_snake2camel(name)]
                elif is_required:
                    # required but missing, let __init__ handle errors
                    types[name] = value_type
            return DataParcel.from_dict(klass.__name__, klass.__annotations__, value, decoders)._data
        JSONConverter.register(klass, encoder, decoder, register_subclasses)
    def __init__(self, klass: type):
        '''
        Initializes a new parcelable class.

        Parameters
        ----------
        klass:
            The type to be wrapped.
        '''
        self.name = klass.__name__
        self.types = klass.__annotations__
        self.__doc__ = klass.__doc__
        JSONConverter.register(
            self,
            encoder=lambda v, e: v.to_dict(e),
            decoder=lambda v, e: DataParcel.from_dict(self.name, self.types, v, e)
        )
    def __repr__(self):
        return f"<parcelable class '{self.name}'>"
    def __call__(self, *args: Any, **kwargs: Any) -> DataParcel:
        for name, value in zip(self.types, args):
            if name in kwargs:
                raise TypeError('{}() got multiple values for argument {!r}'.format(self.name, name))
            kwargs[name] = value
        return DataParcel(self.name, self.types, kwargs)

TypeLike = type | GenericAlias | UnionType | Parcelable | None
'Those who behaves like a type.'

class JSONConverter:
    '''
    Helper class for JSON serialization / deserialization.

    Example
    -------
    >>> def encode_complex(value, encoders):
    ...     return [value.real, value.imag]
    >>> def decode_complex(value, decoders):
    ...     return complex(value[0], value[1])
    >>> JSONConverter.register(complex, encode_complex, decode_complex)
    >>> JSONConverter.encode_json({ 'c': 1+2j }, dict[str, complex])
    '{ "c": [1, 2] }'
    '''
    encoder_registry: dict[TypeLike, Transform] = {}
    ''
    decoder_registry: dict[TypeLike, Transform] = {}
    @classmethod
    def register(cls,
            klass: TypeLike, 
            encoder: Transform | None = None,
            decoder: Transform | None = None,
            register_subclasses: bool = False):
        '''
        Registers a pair of encoder & decoder for a type.

        Parameters
        ----------
        klass:
            The type to register for.
        encoder:
            The encoder for the type.
        decoder:
            The decoder for the type.
        register_subclasses:
            If `True`, all subclasses of the type will be registered.

        Raises
        ------
        TypeError:
            If both `encoder` and `decoder` are `None`.
        '''
        if encoder is None and decoder is None:
            raise TypeError('must specify at least one converter')
        if encoder:
            cls.encoder_registry[klass] = encoder
            if register_subclasses:
                for subclass in klass.__subclasses__():
                    cls.encoder_registry[subclass] = encoder
        if decoder:
            cls.decoder_registry[klass] = decoder
            if register_subclasses:
                for subclass in klass.__subclasses__():
                    cls.decoder_registry[subclass] = decoder
    @classmethod
    def encode(cls, 
               value: Any, 
               value_type: TypeLike, 
               temporary_encoders: Registry = {}) -> Any:
        '''
        Encodes `value` to a JSON-serializable value.

        Parameters
        ----------
        value:
            The value to encode.
        value_type:
            Type of the value.
            Note that this may differ from `type(value)` to provide different encode options.
        temporary_encoders:
            Encoders to use that are not in registry.
        
        Returns
        -------
        obj:
            A JSON-serializable value.
        '''
        if value_type in temporary_encoders:
            return temporary_encoders[value_type](value, temporary_encoders)
        if value_type in cls.encoder_registry:
            return cls.encoder_registry[value_type](value, temporary_encoders)
        if isinstance(value, list):
            if isinstance(value_type, GenericAlias) and value_type.__origin__ is list:
                element_type = value_type.__args__[0]
                result = []
                for element in value:
                    result.append(cls.encode(element, element_type, temporary_encoders))
                return result
            elif isinstance(value_type, UnionType):
                for component in value_type.__args__:
                    try:
                        return cls.encode(value, component, temporary_encoders)
                    except TypeError: pass
            elif value_type is list:
                # no other information, return as-is.
                return value
            raise TypeError('value is list, but value_type is {}'.format(value_type))
        elif isinstance(value, dict):
            if isinstance(value_type, GenericAlias) and value_type.__origin__ is dict:
                # although k_type should be just str, try to decode anyway.
                k_type, v_type = value_type.__args__
                result = {}
                for k, v in value.items():
                    result[cls.encode(k, k_type, temporary_encoders)] = cls.encode(v, v_type, temporary_encoders)
                return result
            elif isinstance(value_type, UnionType):
                for component in value_type.__args__:
                    try:
                        return cls.encode(value, component, temporary_encoders)
                    except TypeError: pass
            elif value_type is dict:
                # no other information, return as-is.
                return value
            raise TypeError('value is dict, but value_type is {}'.format(value_type))
        elif isinstance(value_type, UnionType):
            if len(value_type.__args__) == 2 and NoneType in value_type.__args__:
                if value is None:
                    return None
                else:
                    return cls.encode(value, value_type.__args__[1 - value_type.__args__.index(NoneType)], temporary_encoders)
        # default type, any
        return value
    @classmethod
    def decode(cls, 
               value: Any, 
               value_type: TypeLike,
               temporary_decoders: Registry = {}) -> Any:
        '''
        Decodes `value` to `value_type`.

        Parameters
        ----------
        value:
            The value to decode.
        value_type:
            Type of the decoded value.
        temporary_decoders:
            Decoders to use that are not in registry.
        
        Returns
        -------
        obj:
            The decoded object.
        '''
        if value_type in temporary_decoders:
            return temporary_decoders[value_type](value, temporary_decoders)
        if value_type in cls.decoder_registry:
            return cls.decoder_registry[value_type](value, temporary_decoders)
        if isinstance(value, list):
            if isinstance(value_type, GenericAlias) and value_type.__origin__ is list:
                element_type = value_type.__args__[0]
                result = []
                for element in value:
                    result.append(cls.decode(element, element_type, temporary_decoders))
                return result
            elif isinstance(value_type, UnionType):
                for component in value_type.__args__:
                    try:
                        return cls.decode(value, component, temporary_decoders)
                    except TypeError: pass
            elif value_type is list:
                # no other information, return as-is.
                return value
            raise TypeError('value is list, but value_type is {}'.format(value_type))
        elif isinstance(value, dict):
            if isinstance(value_type, GenericAlias) and value_type.__origin__ is dict:
                # although k_type should be just str, try to decode anyway.
                k_type, v_type = value_type.__args__
                result = {}
                for k, v in value.items():
                    result[cls.decode(k, k_type, temporary_decoders)] = cls.decode(v, v_type, temporary_decoders)
                return result
            elif isinstance(value_type, UnionType):
                for component in value_type.__args__:
                    try:
                        return cls.decode(value, component, temporary_decoders)
                    except TypeError: pass
            elif value_type is dict:
                # no other information, return as-is.
                return value
            raise TypeError('value is dict, but value_type is {}'.format(value_type))
        elif isinstance(value_type, UnionType):
            if len(value_type.__args__) == 2 and NoneType in value_type.__args__:
                if value is None:
                    return None
                else:
                    return cls.decode(value, value_type.__args__[1 - value_type.__args__.index(NoneType)], temporary_decoders)
            # default type, any
        return value
    @classmethod
    def encode_json(cls, 
                    value: Any, 
                    value_type: TypeLike, 
                    temporary_encoders: Registry = {}, 
                    **kwargs: Any) -> str:
        '''
        Encodes `value` to JSON.

        Parameters
        ----------
        value:
            The value to encode.
        value_type:
            Type of the value.
            Note that this may differ from `type(value)` to provide different encode options.
        temporary_encoders:
            Encoders to use that are not in registry.
        
        Returns
        -------
        json:
            JSON representation of `value`.
        '''
        obj = cls.encode(value, value_type, temporary_encoders)
        return json.dumps(obj, **kwargs)
    @classmethod
    def decode_json(cls, 
                    s: str | bytes, 
                    value_type: TypeLike, 
                    temporary_decoders: Registry = {}, 
                    **kwargs: Any) -> Any:
        '''
        Decodes from JSON to `value_type`.

        Parameters
        ----------
        s:
            JSON string to decode from.
        value_type:
            Type of the decoded value.
        temporary_decoders:
            Decoders to use that are not in registry.
        
        Returns
        -------
        obj:
            The decoded object.
        '''
        obj = json.loads(s, **kwargs)
        return cls.decode(obj, value_type, temporary_decoders)

def parse_variable(value: str, default: str | None) -> tuple[str, int]:
    '''
    Parses variable name of form `{prefix}{index}` e.g. `x2`.

    Parameters
    ----------
    value:
        The variable name.
    default:
        Default variable prefix to use if failed to parse.
    
    Returns
    -------
    prefix, index:
        The parsed result.
    '''
    pattern = '^(.+?)(\d+)$'
    matches = re.match(pattern, value)
    if matches is None:
        if default is None:
            raise ValueError(f'failed to parse variable {value!r}')
        return default, 1
    try:
        start = int(matches[2])
    except ValueError:
        start = 1
    return matches[1], start