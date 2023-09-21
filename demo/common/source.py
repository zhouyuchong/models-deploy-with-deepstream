"""
source description
:Author: InSun
:Version: v1.0-alpha
:Created: 2021-07-20
:Description:
"""
import time
__all__ = ["Source", "Stream", "Camera", "IPC"]


class Source(object):
    def __init__(self, id, **kwargs) -> None:
        """
        :param id: str, source id
        :param kwargs:
        """
        self.id = id

        # rt_ctx: dict, runtime context, is used to hold source runtime information
        self.rt_ctx = None

        # name = kwargs.get('name')
        # self.name = name if name and isinstance(name, str) else \
        #     str(id) + '-' + self.__class__.__name__

    # def __str__(self) -> str:
    #     return "<%s %s %s> " % (self.__class__.__name__, self.id, self.name)


class Stream(Source):
    def __init__(self, id, uri, cam=0, ptz_params=None, **kwargs) -> None:
        """
        :param id: str, source id
        :param uri: str, stream uri
        :param kwargs:
        """
        super().__init__(**kwargs)

        self.uri = uri
        self.regions = kwargs.get('regions')
        # idx is the id of uri_decode_bin inside pipeline
        self.idx = None
        self.ptz_params = ptz_params
        self.camera_type = cam
        self.muxid = str(time.time())[:10]

    def set_index(self, index):
        self.idx = index

    def get_index(self):
        return self.idx

    def get_cam_type(self):
        return self.camera_type

    def get_url(self):
        return self.uri

class Camera(Stream):
    def __init__(self, id, uri, cam=0, time=0, value=0, ptz_params=None, **kwargs) -> None:
        """
        :param id: str, source id
        :param uri: str, stream uri
        :param ip: str, camera ip address
        :param kwargs:
            username: str, camera username
            password: str, camera password
        """
        super().__init__(id, uri, cam, ptz_params, **kwargs)
        self.threshold = time
        self.patrol_list = dict()
        self.roi_list = dict()
        if cam:
            self.decode_ptz()


    def decode_url(self): 
        import re
        url = "rtsp://admin:123456@192.168.100.183/video1"
        ip = re.search('admin.*', url)
        ip = ip[0].split('@')
        _ = ip[0].split(':')
        usr_name = _[0]
        pwd = _[1]
        ip = ip[1].split('/')[0]
        print(usr_name, pwd, ip)

    def decode_ptz(self):
        for single_ptz in self.ptz_params:
            tmp_ptz = single_ptz['ptz'].split(',')
            ptz_tuple = (float(tmp_ptz[0]), float(tmp_ptz[1]), float(tmp_ptz[2]))
            self.patrol_list[single_ptz['ptz_id']] = ptz_tuple
        print("ptz list:", self.patrol_list)

    def get_multi_roi(self):
        for single_ptz in self.ptz_params:
            self.roi_list[single_ptz['ptz_id']] = single_ptz['coordinate'][single_ptz['ptz_id']]
        print("roi list: ", self.roi_list)
        return self.roi_list

    def get_threshold(self):
        return self.threshold

    def get_patrol_list(self):
        return self.patrol_list
    

class IPC(Stream):
    def __init__(self, id, uri, ip, **kwargs) -> None:
        """
        :param id: str, source id
        :param uri: str, stream uri
        :param ip: str, camera ip address
        :param kwargs:
            username: str, camera username
            password: str, camera password
        """
        super().__init__(id, uri, **kwargs)
        
        self.ip = ip
        self.username = kwargs.get('username')
        self.password = kwargs.get('password')       
