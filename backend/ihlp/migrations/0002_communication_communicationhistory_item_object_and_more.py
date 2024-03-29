# Generated by Django 4.1.3 on 2022-12-26 08:55

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ihlp', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Communication',
            fields=[
                ('id', models.PositiveBigIntegerField(blank=True, primary_key=True, serialize=False)),
                ('message', models.TextField(blank=True, null=True)),
                ('subject', models.TextField(blank=True, null=True)),
            ],
            options={
                'db_table': 'communication',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='CommunicationHistory',
            fields=[
                ('id', models.PositiveBigIntegerField(blank=True, primary_key=True, serialize=False)),
                ('message', models.TextField(blank=True, null=True)),
                ('subject', models.TextField(blank=True, null=True)),
                ('tblid', models.PositiveBigIntegerField(blank=True, null=True)),
                ('tbltimestamp', models.DateTimeField(blank=True, db_column='tblTimeStamp', null=True)),
            ],
            options={
                'db_table': 'communication_history',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='Item',
            fields=[
                ('id', models.PositiveBigIntegerField(blank=True, primary_key=True, serialize=False, unique=True)),
                ('description', models.TextField(blank=True, null=True)),
                ('subject', models.TextField(blank=True, null=True)),
                ('initials', models.CharField(blank=True, max_length=100, null=True)),
                ('username', models.CharField(blank=True, max_length=100, null=True)),
                ('password', models.CharField(blank=True, max_length=100, null=True)),
                ('firstname', models.CharField(blank=True, max_length=100, null=True)),
                ('lastname', models.CharField(blank=True, max_length=100, null=True)),
                ('email', models.CharField(blank=True, max_length=100, null=True)),
                ('phone', models.CharField(blank=True, max_length=100, null=True)),
                ('ipphone', models.CharField(blank=True, max_length=100, null=True)),
                ('mobile', models.CharField(blank=True, max_length=100, null=True)),
                ('pager', models.CharField(blank=True, max_length=100, null=True)),
                ('fax', models.CharField(blank=True, max_length=100, null=True)),
                ('title', models.CharField(blank=True, max_length=100, null=True)),
                ('sessionid', models.CharField(blank=True, db_column='Sessionid', max_length=100, null=True)),
                ('lastlogon', models.CharField(blank=True, max_length=100, null=True)),
                ('userlanguage', models.CharField(blank=True, db_column='userLanguage', max_length=100, null=True)),
                ('guid', models.CharField(blank=True, max_length=100, null=True)),
            ],
            options={
                'db_table': 'item',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='Object',
            fields=[
                ('id', models.PositiveBigIntegerField(blank=True, primary_key=True, serialize=False)),
                ('externalid', models.PositiveBigIntegerField(blank=True, db_column='externalId', null=True)),
                ('name', models.CharField(blank=True, max_length=100, null=True)),
                ('objecttype', models.CharField(blank=True, db_column='objectType', max_length=100, null=True)),
                ('createdate', models.DateTimeField(blank=True, db_column='createDate', null=True)),
                ('createdby', models.PositiveBigIntegerField(blank=True, db_column='createdBy', null=True)),
                ('altereddate', models.DateTimeField(blank=True, db_column='alteredDate', null=True)),
                ('alteredby', models.PositiveBigIntegerField(blank=True, db_column='alteredBy', null=True)),
                ('state', models.CharField(blank=True, max_length=100, null=True)),
                ('metatype', models.CharField(blank=True, db_column='metaType', max_length=100, null=True)),
                ('syncid', models.CharField(blank=True, max_length=100, null=True)),
                ('indicatorchangedate', models.DateTimeField(blank=True, null=True)),
                ('enterpriceroot', models.CharField(blank=True, max_length=100, null=True)),
            ],
            options={
                'db_table': 'object',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='ObjectHistory',
            fields=[
                ('id', models.PositiveBigIntegerField(blank=True, primary_key=True, serialize=False)),
                ('externalid', models.PositiveBigIntegerField(blank=True, db_column='externalId', null=True)),
                ('name', models.CharField(blank=True, max_length=100, null=True)),
                ('objecttype', models.CharField(blank=True, db_column='objectType', max_length=100, null=True)),
                ('createdate', models.DateTimeField(blank=True, db_column='createDate', null=True)),
                ('createdby', models.PositiveBigIntegerField(blank=True, db_column='createdBy', null=True)),
                ('altereddate', models.DateTimeField(blank=True, db_column='alteredDate', null=True)),
                ('alteredby', models.PositiveBigIntegerField(blank=True, db_column='alteredBy', null=True)),
                ('state', models.CharField(blank=True, max_length=100, null=True)),
                ('tblid', models.PositiveBigIntegerField(blank=True, null=True)),
                ('owntimestamp', models.DateTimeField(blank=True, db_column='ownTimeStamp', null=True)),
                ('metatype', models.CharField(blank=True, db_column='metaType', max_length=100, null=True)),
                ('indicatorchangedate', models.DateTimeField(blank=True, null=True)),
                ('enterpriceroot', models.CharField(blank=True, max_length=100, null=True)),
            ],
            options={
                'db_table': 'object_history',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='Relation',
            fields=[
                ('id', models.PositiveBigIntegerField(blank=True, primary_key=True, serialize=False)),
                ('leftid', models.PositiveBigIntegerField(blank=True, db_column='leftId', null=True)),
                ('rightid', models.PositiveBigIntegerField(blank=True, db_column='rightId', null=True)),
                ('relationtypeid', models.IntegerField(blank=True, db_column='relationTypeID', null=True)),
                ('lefttype', models.CharField(blank=True, db_column='leftType', max_length=100, null=True)),
                ('righttype', models.CharField(blank=True, db_column='rightType', max_length=100, null=True)),
            ],
            options={
                'db_table': 'relation',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='RelationExportTasktype',
            fields=[
                ('id', models.PositiveBigIntegerField(blank=True, primary_key=True, serialize=False)),
                ('leftid', models.PositiveBigIntegerField(blank=True, db_column='leftId', null=True)),
                ('rightid', models.PositiveBigIntegerField(blank=True, db_column='rightId', null=True)),
                ('relationtypeid', models.IntegerField(blank=True, db_column='relationTypeID', null=True)),
                ('lefttype', models.CharField(blank=True, db_column='leftType', max_length=100, null=True)),
                ('righttype', models.CharField(blank=True, db_column='rightType', max_length=100, null=True)),
            ],
            options={
                'db_table': 'relation_export_tasktype',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='RelationHistory',
            fields=[
                ('id', models.PositiveBigIntegerField(blank=True, primary_key=True, serialize=False)),
                ('leftid', models.PositiveBigIntegerField(blank=True, db_column='leftId', null=True)),
                ('rightid', models.PositiveBigIntegerField(blank=True, db_column='rightId', null=True, unique=True)),
                ('relationtypeid', models.IntegerField(blank=True, db_column='relationTypeID', null=True)),
                ('lefttype', models.CharField(blank=True, db_column='leftType', max_length=100, null=True)),
                ('righttype', models.CharField(blank=True, db_column='rightType', max_length=100, null=True)),
                ('tblid', models.IntegerField(blank=True, null=True, unique=True)),
                ('tbltimestamp', models.DateTimeField(blank=True, db_column='tblTimeStamp', null=True)),
            ],
            options={
                'db_table': 'relation_history',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='Request',
            fields=[
                ('id', models.PositiveBigIntegerField(blank=True, primary_key=True, serialize=False, unique=True)),
                ('description', models.TextField(blank=True, null=True)),
                ('itilstate', models.CharField(blank=True, db_column='itilState', max_length=100, null=True)),
                ('receiveddate', models.DateTimeField(blank=True, db_column='receivedDate', null=True)),
                ('receivedvia', models.CharField(blank=True, db_column='receivedVia', max_length=100, null=True)),
                ('servicedisorderdate', models.DateTimeField(blank=True, db_column='serviceDisorderDate', null=True)),
                ('automaticallycreated', models.CharField(blank=True, db_column='automaticallyCreated', max_length=100, null=True)),
                ('callbackmethod', models.CharField(blank=True, db_column='callBackMethod', max_length=100, null=True)),
                ('cost', models.CharField(blank=True, max_length=100, null=True)),
                ('timeconsumption', models.CharField(blank=True, db_column='timeConsumption', max_length=100, null=True)),
                ('numberofalarms', models.CharField(blank=True, db_column='numberOfAlarms', max_length=100, null=True)),
                ('subject', models.CharField(blank=True, max_length=100, null=True)),
                ('classification', models.CharField(blank=True, max_length=100, null=True)),
                ('priority', models.CharField(blank=True, max_length=100, null=True)),
                ('impact', models.CharField(blank=True, max_length=100, null=True)),
                ('urgency', models.CharField(blank=True, max_length=100, null=True)),
                ('deadline', models.DateTimeField(blank=True, null=True)),
                ('investigationanddiagnosis', models.CharField(blank=True, db_column='investigationAndDiagnosis', max_length=100, null=True)),
                ('cause', models.CharField(blank=True, max_length=100, null=True)),
                ('workaround', models.CharField(blank=True, db_column='workAround', max_length=100, null=True)),
                ('solution', models.TextField(blank=True, null=True)),
                ('solutiondate', models.DateTimeField(blank=True, db_column='solutionDate', null=True)),
                ('closingcode', models.CharField(blank=True, db_column='closingCode', max_length=100, null=True)),
                ('expectedsolutiondatetime', models.DateTimeField(blank=True, db_column='expectedSolutionDateTime', null=True)),
                ('impactifimplemented', models.CharField(blank=True, db_column='impactIfImplemented', max_length=100, null=True)),
                ('impactifnotimplemented', models.CharField(blank=True, db_column='impactIfNotImplemented', max_length=100, null=True)),
                ('expectedresourcecost', models.CharField(blank=True, db_column='expectedResourceCost', max_length=100, null=True)),
                ('approvalofcontentoftherfc', models.CharField(blank=True, db_column='approvalOfContentOfTheRFC', max_length=100, null=True)),
                ('starttime', models.DateTimeField(blank=True, null=True)),
                ('endtime', models.DateTimeField(blank=True, null=True)),
                ('currentsection', models.CharField(blank=True, db_column='currentSection', max_length=100, null=True)),
                ('previoussections', models.CharField(blank=True, db_column='previousSections', max_length=100, null=True)),
                ('permanentlycloseddate', models.DateTimeField(blank=True, null=True)),
            ],
            options={
                'db_table': 'request',
                'managed': False,
            },
        ),
        migrations.AddField(
            model_name='predict',
            name='request_id',
            field=models.IntegerField(default=0),
            preserve_default=False,
        ),
    ]
