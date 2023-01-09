import os
import time

import numpy as np
import pandas as pd

from helpers.bulkinsert import BulkCreateManager
from ihlp.models_ihlp import Request, Item, RelationHistory, ObjectHistory, Relation, Object

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

class Boot:

    def __init__(self, debug=True):
        tic = time.perf_counter()
        toc = time.perf_counter()
        print(f"Booted in {toc - tic:0.4f} seconds")

    def users(self):
        pass

    def reset(self):
        pass

    def load(self):

        if Request.objects.using('ihlp').count() == 0:
            PATH = BASE_DIR + '/notebooks/database/'
            requests = pd.read_csv(PATH + 'Request.csv', encoding='UTF-8', delimiter=';', quotechar='"', dtype=str)
            requests = requests.fillna(np.nan).replace([np.nan], [None])
            bulk_mgr = BulkCreateManager(chunk_size=50)
            for _, el in requests.iterrows():
                bulk_mgr.add(Request(
                    id=el.id,
                    description=el.description,
                    itilstate=el.itilState,
                    receiveddate=el.receivedDate,
                    receivedvia=el.receivedVia,
                    servicedisorderdate=el.serviceDisorderDate,
                    automaticallycreated=el.automaticallyCreated,
                    callbackmethod=el.callBackMethod,
                    cost=el.cost,
                    timeconsumption=el.timeConsumption,
                    numberofalarms=el.numberOfAlarms,
                    subject=el.subject,
                    classification=el.classification,
                    priority=el.priority,
                    impact=el.impact,
                    urgency=el.urgency,
                    deadline=el.deadline,
                    investigationanddiagnosis=el.investigationAndDiagnosis,
                    cause=el.cause,
                    workaround=el.workAround,
                    solution=el.solution,
                    solutiondate=el.solutionDate,
                    closingcode=el.closingCode,
                    expectedsolutiondatetime=el.expectedSolutionDateTime,
                    impactifimplemented=el.impactIfImplemented,
                    impactifnotimplemented=el.impactIfNotImplemented,
                    expectedresourcecost=el.expectedResourceCost,
                    approvalofcontentoftherfc=el.approvalOfContentOfTheRFC,
                    starttime=el.starttime,
                    endtime=el.endtime,
                    currentsection=el.currentSection,
                    previoussections=el.previousSections,
                    permanentlycloseddate=el.permanentlycloseddate
                ))
            bulk_mgr.done()

        if Item.objects.using('ihlp').count() == 0:
            PATH = BASE_DIR + '/notebooks/database/'
            items = pd.read_csv(PATH + 'Item.csv', encoding='UTF-8', delimiter=';', quotechar='"', dtype=str)
            items = items.fillna(np.nan).replace([np.nan], [None])
            bulk_mgr = BulkCreateManager(chunk_size=50)
            for _, el in items.iterrows():
                bulk_mgr.add(Item(
                    id=el.id,
                    description=el.description,
                    subject=el.subject,
                    initials=el.initials,
                    username=el.username,
                    password=el.password,
                    firstname=el.firstname,
                    lastname=el.lastname,
                    email=el.email,
                    phone=el.phone,
                    ipphone=el.ipphone,
                    mobile=el.mobile,
                    pager=el.pager,
                    fax=el.fax,
                    title=el.title,
                    sessionid=el.Sessionid,
                    lastlogon=el.lastlogon,
                    userlanguage=el.userLanguage,
                    guid=el.guid
                ))
            bulk_mgr.done()

        if Relation.objects.using('ihlp').count() == 0:
            PATH = BASE_DIR + '/notebooks/database/'
            relations = pd.read_csv(PATH + 'Relation.csv', encoding='UTF-8', delimiter=';', quotechar='"', dtype=str)
            relations = relations.fillna(np.nan).replace([np.nan], [None])
            bulk_mgr = BulkCreateManager(chunk_size=50)
            for _, el in relations.iterrows():
                bulk_mgr.add(Relation(
                    id=el.id,
                    leftid=el.leftID,
                    rightid=el.rightID,
                    relationtypeid=el.relationTypeID,
                    lefttype=el.leftType,
                    righttype=el.rightType
                ))
            bulk_mgr.done()

        if Object.objects.using('ihlp').count() == 0:
            PATH = BASE_DIR + '/notebooks/database/'
            objects = pd.read_csv(PATH + 'Object.csv', encoding='UTF-8', delimiter=';', dtype=str)
            objects = objects.fillna(np.nan).replace([np.nan], [None])
            bulk_mgr = BulkCreateManager(chunk_size=50)
            for _, el in objects.iterrows():
                bulk_mgr.add(Object(
                    id=el.id,
                    externalid=el.externalId,
                    name=el.values[2],  # el.name is an int, no idea why.
                    objecttype=el.objectType,
                    createdate=el.createDate,
                    createdby=el.createdBy,
                    altereddate=el.alteredDate,
                    alteredby=el.alteredBy,
                    state=el.state,
                    metatype=el.metaType,
                    syncid=el.syncid,
                    indicatorchangedate=el.indicatorchangedate,
                    enterpriceroot=el.enterpriceroot
                ))
            bulk_mgr.done()