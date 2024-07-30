


def get_title(layout):
    t = layout
    if t.get('title'):
        return t['title']
    else:
        return None

def get_barmode(layout):
    t = layout
    if t.get('barmode'):
        return t['barmode']
    else:
        return None

def get_width(layout):
    t = layout
    if t.get('width'):
        return t['width']
    else:
        return None

def get_height(layout):
    t = layout
    if t.get('height'):
        return t['height']
    else:
        return None

def get_xrange(layout):
    t = layout
    if t.get('xaxis'):
        if t.get('xaxis').get('range'):
            return t['xaxis']['range']
        else:
            return None
    else:
        return None

def get_yrange(layout):
    t = layout
    if t.get('yaxis'):
        if t.get('yaxis').get('range'):
            return t['yaxis']['range']
        else:
            return None
    else:
        return None

def get_xtitle(layout):
    t = layout
    if t.get('xaxis'):
        if t.get('xaxis').get('title'):
            return t['xaxis']['title']
        else:
            return None
    else:
        return None

def get_ytitle(layout):
    t = layout
    if t.get('yaxis'):
        if t.get('yaxis').get('title'):
            return t['yaxis']['title']
        else:
            return None
    else:
        return None
    
def get_ztitle(layout):
    t = layout
    if t.get('zaxis'):
        if t.get('zaxis').get('title'):
            return t['zaxis']['title']
        else:
            return None
    else:
        return None