def get_volume(geometry_id):
    """ Returns the volume information of a geometry ID object provided.
        E.g.: get_volume(936748859932016651) = 0d """
    gid = hex(int(geometry_id))[2:]
    gid = '0' * (16 - len(gid)) + gid
    return int(gid[-16:-14], base=16)

def get_boundary(geometry_id):
    """ Returns the boundary information of a geometry ID object provided.
        E.g.: get_volume('936748859932016651') = 000 """
    gid = hex(int(geometry_id))[2:]
    gid = '0' * (16 - len(gid)) + gid
    return int(gid[-14:-12], base=16)

def get_layer(geometry_id):
    """ Returns the layer information of a geometry ID object provided.
        E.g.: get_volume('936748859932016651') = 002 """
    gid = hex(int(geometry_id))[2:]
    gid = '0' * (16 - len(gid)) + gid
    return int(gid[-12:-9], base=16)

def get_approach(geometry_id):
    """ Returns the approach information of a geometry ID object provided.
        E.g.: get_volume('936748859932016651') = 00 """
    gid = hex(int(geometry_id))[2:]
    gid = '0' * (16 - len(gid)) + gid
    return int(gid[-9:-7], base=16)

def get_sensitivity(geometry_id):
    """ Returns the sensitivity information of a geometry ID object provided.
        E.g.: get_volume('936748859932016651') = 000000b """
    gid = hex(int(geometry_id))[2:]
    gid = '0' * (16 - len(gid)) + gid
    return int(gid[-7:], base=16)