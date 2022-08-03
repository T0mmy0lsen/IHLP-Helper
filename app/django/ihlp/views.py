from django.http import HttpResponse
from django.shortcuts import render

# Create your views here.


def get_request(request, request_id):
    return HttpResponse([1, 2, 3, 4])