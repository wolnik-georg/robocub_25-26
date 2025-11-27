#!/usr/bin/env python
""" An enhanced Interactive Python for NAO
"""

__author__ = 'Yuan Xu'

from naoqi import ALProxy


class IProxy(ALProxy):
    '''Interactive proxy to ALModule'''
    def __init__(self, name, ip, port):
        ALProxy.__init__(self, name, ip, port)
        methods = self.call('getMethodList')

        def createMethod(metholdHelp):
            m, description, parameters, returnName, returnDescription = metholdHelp

            def method(*args, **kwargs):
                '''method of proxy'''
                try:
                    ret = self.call(m, *args, **kwargs)
                except RuntimeError, e:
                    raise e
                return ret

            def postMethod(*args, **kwargs):
                '''post method of proxy'''
                try:
                    ret = self.pCall(m, *args, **kwargs)
                except RuntimeError, e:
                    raise e
                return ret
            # pretty doc
            method_line = m + '(' + ', '.join([n for n, d in parameters]) + ')'
            if returnName:
                method_line += ' -> ' + returnName
            method_line += '\n' + '-' * len(method_line)
            parameters = '\n'.join([n + " : " + d for n, d in parameters])
            if parameters:
                parameters = '\nParameters\n----------\n' + parameters
            if returnName or returnDescription:
                return_str = ('\nReturns\n-------\n' + returnName + " : " + returnDescription)
            else:
                return_str = ''
            method.__doc__ = method_line + '\n' + description + '\n' + parameters + '\n' + return_str
            postMethod.__doc__ = method.__doc__
            return method, postMethod

        for m in methods:
            metholdHelp = self.call('getMethodHelp', m)
            methodName = metholdHelp[0]
            if methodName == m or not hasattr(self, methodName):
                method, postMethod = createMethod(metholdHelp)
                setattr(self, methodName, method)
                setattr(self.post, methodName, postMethod)
            else:
                doc = self.call('getUsage', m)
                f = getattr(self, methodName)
                pf = getattr(self.post, methodName)
                if f.__doc__:
                    doc = doc.split('\n')
                    f.__doc__ += '\n'.join(doc[2:])
                else:
                    f.__doc__ = doc
                pf.__doc__ = f.__doc__
                setattr(self, methodName, f)
                setattr(self.post, methodName, pf)
        moduleHelp = self.getModuleHelp()
        if moduleHelp and moduleHelp[0]:
            self.__doc__ = moduleHelp[0]
            if moduleHelp[1]:
                for example in moduleHelp[1]:
                    self.__doc__ += '\n' + '-' * 20 + ' Example in ' + example[0] + ' ' + '-' * 20
                    self.__doc__ += '\n' + example[1] + '\n'

        self.post.__doc__ = 'The post object of ' + name + ''' proxy. It enables
            you to do other work at the same time (e.g. walking while talking).
            Each post call generates a task id. You can use this task id to
            check if a task is running, or wait until the task is finished.'''


class NAO:
    '''A class has some useful proxies to NAO'''
    def __init__(self, broker, port=9559):
        self._broker = broker
        self._port = port

        memory = self.createProxy('ALMemory', raiseErrors=True)
        if memory:
            self.memory = memory

        motion = self.createProxy('ALMotion')
        if motion:
            self.motion = motion

        tts = self.createProxy('ALTextToSpeech')
        if tts:
            self.tts = tts

        leds = self.createProxy('ALLeds')
        if leds:
            self.leds = leds

        vision = self.createProxy('RobocupVision')
        if vision:
            self.vision = vision

        agent = self.createProxy('DAInamiteAgent')
        if agent:
            self.agent = agent

    def createProxy(self, name, raiseErrors=False):
        '''create a proxy to a module in this NAO'''
        try:
            proxy = IProxy(name, self._broker, self._port)
            return proxy
        except RuntimeError:
            print '[WARN] ', name, ' is not running!'
            if raiseErrors:
                raise
