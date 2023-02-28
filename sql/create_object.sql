-- ihlp.`object` definition

CREATE TABLE `object` (
  `id` bigint(20) unsigned DEFAULT NULL,
  `externalId` bigint(20) unsigned DEFAULT NULL,
  `name` varchar(100) DEFAULT NULL,
  `objectType` varchar(100) DEFAULT NULL,
  `createDate` timestamp NULL DEFAULT NULL,
  `createdBy` bigint(20) unsigned DEFAULT NULL,
  `alteredDate` timestamp NULL DEFAULT NULL,
  `alteredBy` bigint(20) unsigned DEFAULT NULL,
  `state` varchar(100) DEFAULT NULL,
  `metaType` varchar(100) DEFAULT NULL,
  `syncid` varchar(100) DEFAULT NULL,
  `indicatorchangedate` timestamp NULL DEFAULT NULL,
  `enterpriceroot` varchar(100) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;