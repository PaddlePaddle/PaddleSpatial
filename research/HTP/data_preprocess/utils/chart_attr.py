from utils.detect_type import detect_field_type, data_type_to_general_type, check_if_line_chart 


def get_add_attr(series_type, series_data):
    
    width, color, bins_num, size = None, None, None, None
    
    if series_type == 'line':
        width, color = get_line_attr(series_data)
    elif series_type == 'box':
        color = get_box_attr(series_data)
    elif series_type == 'histogram':
        color,bins_num = get_his_attr(series_data)
    elif series_type == 'bar':
        color = get_bar_attr(series_data)
    elif series_type == 'scatter':
        color,size = get_scatter_attr(series_data)
        
    return width, color, bins_num,size

def get_xsrc(t, series_type):
    if series_type == 'pie':
        xsrc = t.get('labelssrc')
    else:
        xsrc = t.get('xsrc')
    if xsrc:
        xsrc = str(xsrc)
        xsrc = xsrc.split(':')[0]+':'+xsrc.split(':')[2]
    else:
        xsrc = None
    return xsrc
def get_ysrc(t,series_type):
    if series_type == 'pie':
        ysrc = t.get('valuessrc')
    else:
        ysrc = t.get('ysrc')
    if ysrc:
        ysrc = str(ysrc)
        ysrc = ysrc.split(':')[0]+':'+ysrc.split(':')[2]
    else:
        ysrc = None
    return ysrc

def get_zsrc(t, series_type):
    zsrc = t.get('zsrc')
    if zsrc:
        zsrc = str(zsrc)
        zsrc = zsrc.split(':')[0]+':'+zsrc.split(':')[2]
    else:
        zsrc = None
    return zsrc

def get_type(t):
    chart_type = t.get('type')
    if chart_type:
        if chart_type == 'scatter' and check_if_line_chart(t):
            chart_type = 'line'
        return chart_type
    else:
        return None

def get_series_name(t):
    return t.get('name') if t.get('name') else None

def get_box_attr(box_data):
    box_attrs = box_data.get('marker')
    try:
        if box_attrs:
            color = box_attrs.get('color') if box_attrs.get('color') else None
            return color
        else: 
            return None
    except:
        return None

def get_his_attr(his_data):
    bins_num = None
    color = None
    marker = his_data.get('marker')
    if marker:
        color = marker.get('color') if marker.get('color') else None
    bins = his_data.get('xbins')
    
    if bins:
        try:
            if bins.get('start') and bins.get('size') and bins.get('end'):
                start = float(bins.get('start'))
                size = float(bins.get('size'))
                if size != 0:
                    end = float(bins.get('end'))
                    bins_num = float((end-start)/size) 
                    if not bins_num>0:
                        bins_num = None
        except:
            bins_num = None
    return color,bins_num
    
def get_bar_attr(bar_data):
    bar_attrs = bar_data.get('marker')
    try:
        if bar_attrs:
            color = bar_attrs.get('color') if bar_attrs.get('color') else None
            return color
        else: 
            return None
    except:
        None

def get_scatter_attr(scatter_data):
    scatter_attrs = scatter_data.get('marker')
    try:
        if scatter_attrs:
            color = scatter_attrs.get('color') if scatter_attrs.get('color') else None
            size  = scatter_attrs.get('size') if scatter_attrs.get('size') else None
            return color, size
        else: 
            return None,None
    except:
        return None,None

def get_line_attr(line_data):
    line_attrs = line_data.get('line')
    try:
        if line_attrs:
            width = line_attrs.get('width') if line_attrs.get('width') else None
            color = line_attrs.get('color') if line_attrs.get('color') else None
            return width,color
        else: 
            return None,None
    except:
        return None,None
