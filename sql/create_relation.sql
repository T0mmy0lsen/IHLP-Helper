-- ihlp.relation definition

CREATE TABLE `relation` (
  `id` bigint(20) unsigned DEFAULT NULL,
  `leftId` bigint(20) unsigned DEFAULT NULL,
  `rightId` bigint(20) unsigned DEFAULT NULL,
  `relationTypeID` int(11) DEFAULT NULL,
  `leftType` varchar(100) DEFAULT NULL,
  `rightType` varchar(100) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;