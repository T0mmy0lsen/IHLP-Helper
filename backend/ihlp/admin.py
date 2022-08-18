from django.contrib import admin

from ihlp.models import Schedule, Slot, Machine, Predict

admin.site.register(Schedule)
admin.site.register(Predict)
admin.site.register(Machine)
admin.site.register(Slot)
