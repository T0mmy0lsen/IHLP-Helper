# Generated by Django 4.1.3 on 2022-12-26 15:57

from django.db import migrations, models
import jsonfield.fields


class Migration(migrations.Migration):

    dependencies = [
        ('ihlp', '0002_communication_communicationhistory_item_object_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='Workload',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('request_id', models.IntegerField()),
                ('data', jsonfield.fields.JSONField()),
            ],
        ),
    ]