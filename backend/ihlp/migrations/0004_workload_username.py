# Generated by Django 4.1.3 on 2022-12-26 16:50

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ihlp', '0003_workload'),
    ]

    operations = [
        migrations.AddField(
            model_name='workload',
            name='username',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
    ]
