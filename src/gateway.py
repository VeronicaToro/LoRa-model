#!/usr/bin/env python3
# -*- coding: utf-8 -*-
class Gateway(object):
    def __init__(self, id, posX, posY, radius):
        self._id = id
        self._posX = posX
        self._posY = posY
        self._radius = radius

    def get_location(self):
        return self._posX, self._posY

    def get_radius(self):
        return self._radius
