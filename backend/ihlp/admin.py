from django.contrib import admin
from ihlp.models_ihlp import Request
from ihlp.models import Predict

admin.site.register(Predict)
admin.site.register(Request)
