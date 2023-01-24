# Generated by Django 4.1.5 on 2023-01-22 12:53

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ihlp', '0004_workload_username'),
    ]

    operations = [
        migrations.CreateModel(
            name='Feedback',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('request_id', models.IntegerField()),
                ('type', models.CharField(blank=True, max_length=100, null=True)),
                ('message', models.CharField(blank=True, max_length=2000, null=True)),
            ],
        ),
    ]
