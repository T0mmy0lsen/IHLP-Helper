-- ihlp.item definition

CREATE TABLE `item` (
  `id` bigint(20) NOT NULL,
  `description` longtext DEFAULT NULL,
  `subject` longtext DEFAULT NULL,
  `initials` varchar(30) CHARACTER SET utf8 DEFAULT NULL,
  `username` varchar(50) CHARACTER SET utf8 DEFAULT NULL,
  `password` varchar(50) CHARACTER SET utf8 DEFAULT NULL,
  `firstname` varchar(255) CHARACTER SET utf8 DEFAULT NULL,
  `lastname` varchar(255) CHARACTER SET utf8 DEFAULT NULL,
  `email` varchar(255) CHARACTER SET utf8 DEFAULT NULL,
  `phone` varchar(50) CHARACTER SET utf8 DEFAULT NULL,
  `ipphone` varchar(50) CHARACTER SET utf8 DEFAULT NULL,
  `mobile` varchar(50) CHARACTER SET utf8 DEFAULT NULL,
  `pager` varchar(50) CHARACTER SET utf8 DEFAULT NULL,
  `fax` varchar(50) CHARACTER SET utf8 DEFAULT NULL,
  `title` varchar(255) CHARACTER SET utf8 DEFAULT NULL,
  `Sessionid` varchar(50) CHARACTER SET utf8 DEFAULT NULL,
  `lastlogon` datetime DEFAULT NULL,
  `userLanguage` varchar(5) CHARACTER SET utf8 DEFAULT NULL,
  `guid` varchar(50) CHARACTER SET utf8 DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;