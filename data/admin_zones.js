/*
Dictionary format
district -> upazila -> pcode
*/
const admin_zones = {
  Bagerhat: {
    "Bagerhat Sadar": 0,
    Chitalmari: 1,
    Fakirhat: 2,
    Kachua: 3,
    Mollahat: 4,
    Mongla: 5,
    Morrelganj: 6,
    Rampal: 7,
    Sarankhola: 8,
  },
  Bandarban: {
    Alikadam: 9,
    "Bandarban Sadar": 10,
    Lama: 11,
    Naikhongchhari: 12,
    Rowangchhari: 13,
    Ruma: 14,
    Thanchi: 15,
  },
  Barguna: {
    Amtali: 16,
    Bamna: 17,
    "Barguna Sadar": 18,
    Betagi: 19,
    Patharghata: 20,
  },
  Barisal: {
    Agailjhara: 21,
    Babuganj: 22,
    Bakerganj: 23,
    "Banari Para": 24,
    "Barisal Sadar (Kotwali)": 25,
    Gaurnadi: 26,
    Hizla: 27,
    Mehendiganj: 28,
    Muladi: 29,
    Wazirpur: 30,
  },
  Bhola: {
    "Bhola Sadar": 31,
    Burhanuddin: 32,
    "Char Fasson": 33,
    Daulatkhan: 34,
    Lalmohan: 35,
    Manpura: 36,
    Tazumuddin: 37,
  },
  Bogra: {
    Adamdighi: 38,
    "Bogra Sadar": 39,
    Dhunat: 40,
    Dhupchanchia: 41,
    Gabtali: 42,
    Kahaloo: 43,
    Nandigram: 44,
    Sariakandi: 45,
    Shajahanpur: 46,
    Sherpur: 47,
    Shibganj: 48,
    Sonatola: 49,
  },
  Brahamanbaria: {
    Akhaura: 50,
    Ashuganj: 51,
    Banchharampur: 52,
    Bijoynagar: 53,
    "Brahmanbaria Sadar": 54,
    Kasba: 55,
    Nabinagar: 56,
    Nasirnagar: 57,
    Sarail: 58,
  },
  Chandpur: {
    "Chandpur Sadar": 59,
    Faridganj: 60,
    "Haim Char": 61,
    Hajiganj: 62,
    Kachua: 63,
    "Matlab Dakshin": 64,
    "Matlab Uttar": 65,
    Shahrasti: 66,
  },
  Chittagong: {
    Anowara: 67,
    Bakalia: 68,
    Banshkhali: 69,
    "Bayejid Bostami": 70,
    Boalkhali: 71,
    Chandanaish: 72,
    Chandgaon: 73,
    "Chittagong Port": 74,
    "Double Mooring": 75,
    Fatikchhari: 76,
    Halishahar: 77,
    Hathazari: 78,
    Khulshi: 79,
    Kotwali: 80,
    Lohagara: 81,
    Mirsharai: 82,
    Pahartali: 83,
    Panchlaish: 84,
    Patenga: 85,
    Patiya: 86,
    Rangunia: 87,
    Raozan: 88,
    Sandwip: 89,
    Satkania: 90,
    Sitakunda: 91,
  },
  Chuadanga: {
    Alamdanga: 92,
    "Chuadanga Sadar": 93,
    Damurhuda: 94,
    "Jiban Nagar": 95,
  },
  Comilla: {
    Barura: 96,
    "Brahman Para": 97,
    Burichang: 98,
    Chandina: 99,
    Chauddagram: 100,
    "Comilla Adarsha Sadar": 101,
    "Comilla Sadar Dakshin": 102,
    Daudkandi: 103,
    Debidwar: 104,
    Homna: 105,
    Laksam: 106,
    Manoharganj: 107,
    Meghna: 108,
    Muradnagar: 109,
    Nangalkot: 110,
    Titas: 111,
  },
  "Cox's Bazar": {
    Chakaria: 112,
    "Cox's Bazar Sadar": 113,
    Kutubdia: 114,
    Maheshkhali: 115,
    Pekua: 116,
    Ramu: 117,
    Teknaf: 118,
    Ukhia: 119,
  },
  Dhaka: {
    Adabor: 120,
    Badda: 121,
    Bangshal: 122,
    "Biman Bandar": 123,
    Cantonment: 124,
    "Chak Bazar": 125,
    Dakshinkhan: 126,
    "Darus Salam": 127,
    Demra: 128,
    Dhamrai: 129,
    Dhanmondi: 130,
    Dohar: 131,
    Gendaria: 132,
    Gulshan: 133,
    Hazaribagh: 134,
    Jatrabari: 135,
    Kadamtali: 136,
    Kafrul: 137,
    Kalabagan: 138,
    "Kamrangir Char": 139,
    Keraniganj: 140,
    Khilgaon: 141,
    Khilkhet: 142,
    Kotwali: 143,
    Lalbagh: 144,
    Mirpur: 145,
    Mohammadpur: 146,
    Motijheel: 147,
    Nawabganj: 148,
    "New Market": 149,
    Pallabi: 150,
    Paltan: 151,
    Ramna: 152,
    Rampura: 153,
    Sabujbagh: 154,
    Savar: 155,
    "Shah Ali": 156,
    Shahbagh: 157,
    "Sher-e-bangla Nagar": 158,
    Shyampur: 159,
    Sutrapur: 160,
    Tejgaon: 161,
    "Tejgaon Ind. Area": 162,
    Turag: 163,
    "Uttar Khan": 164,
    Uttara: 165,
  },
  Dinajpur: {
    Biral: 166,
    Birampur: 167,
    Birganj: 168,
    Bochaganj: 169,
    Chirirbandar: 170,
    "Dinajpur Sadar": 171,
    Fulbari: 172,
    Ghoraghat: 173,
    Hakimpur: 174,
    Kaharole: 175,
    Khansama: 176,
    Nawabganj: 177,
    Parbatipur: 178,
  },
  Faridpur: {
    Alfadanga: 179,
    Bhanga: 180,
    Boalmari: 181,
    "Char Bhadrasan": 182,
    "Faridpur Sadar": 183,
    Madhukhali: 184,
    Nagarkanda: 185,
    Sadarpur: 186,
    Saltha: 187,
  },
  Feni: {
    Chhagalnaiya: 188,
    Daganbhuiyan: 189,
    "Feni Sadar": 190,
    Fulgazi: 191,
    Parshuram: 192,
    Sonagazi: 193,
  },
  Gaibandha: {
    Fulchhari: 194,
    "Gaibandha Sadar": 195,
    Gobindaganj: 196,
    Palashbari: 197,
    Sadullapur: 198,
    Saghatta: 199,
    Sundarganj: 200,
  },
  Gazipur: {
    "Gazipur Sadar": 201,
    Kaliakair: 202,
    Kaliganj: 203,
    Kapasia: 204,
    Sreepur: 205,
  },
  Gopalganj: {
    "Gopalganj Sadar": 206,
    Kashiani: 207,
    "Kotali Para": 208,
    Muksudpur: 209,
    "Tungi Para": 210,
  },
  Habiganj: {
    Ajmiriganj: 211,
    Bahubal: 212,
    Baniachong: 213,
    Chunarughat: 214,
    "Habiganj Sadar": 215,
    Lakhai: 216,
    Madhabpur: 217,
    Nabiganj: 218,
  },
  Jamalpur: {
    Bakshiganj: 219,
    Dewanganj: 220,
    Islampur: 221,
    "Jamalpur Sadar": 222,
    Madarganj: 223,
    Melandaha: 224,
    Sarishabari: 225,
  },
  Jessore: {
    Abhaynagar: 226,
    "Bagher Para": 227,
    Chaugachha: 228,
    Jhikargachha: 229,
    Keshabpur: 230,
    Kotwali: 231,
    Manirampur: 232,
    Sharsha: 233,
  },
  Jhalokati: {
    "Jhalokati Sadar": 234,
    Kanthalia: 235,
    Nalchity: 236,
    Rajapur: 237,
  },
  Jhenaidah: {
    Harinakunda: 238,
    "Jhenaidah Sadar": 239,
    Kaliganj: 240,
    Kotchandpur: 241,
    Maheshpur: 242,
    Shailkupa: 243,
  },
  Joypurhat: {
    Akkelpur: 244,
    "Joypurhat Sadar": 245,
    Kalai: 246,
    Khetlal: 247,
    Panchbibi: 248,
  },
  Khagrachhari: {
    Dighinala: 249,
    "Khagrachhari Sadar": 250,
    Lakshmichhari: 251,
    Mahalchhari: 252,
    Manikchhari: 253,
    Matiranga: 254,
    Panchhari: 255,
    Ramgarh: 256,
  },
  Khulna: {
    Batiaghata: 257,
    Dacope: 258,
    Daulatpur: 259,
    Dighalia: 260,
    Dumuria: 261,
    Khalishpur: 262,
    "Khan Jahan Ali": 263,
    "Khulna Sadar": 264,
    Koyra: 265,
    Paikgachha: 266,
    Phultala: 267,
    Rupsa: 268,
    Sonadanga: 269,
    Terokhada: 270,
  },
  Kishoreganj: {
    Austagram: 271,
    Bajitpur: 272,
    Bhairab: 273,
    Hossainpur: 274,
    Itna: 275,
    Karimganj: 276,
    Katiadi: 277,
    "Kishoreganj Sadar": 278,
    "Kuliar Char": 279,
    Mithamain: 280,
    Nikli: 281,
    Pakundia: 282,
    Tarail: 283,
  },
  Kurigram: {
    Bhurungamari: 284,
    "Char Rajibpur": 285,
    Chilmari: 286,
    "Kurigram Sadar": 287,
    Nageshwari: 288,
    Phulbari: 289,
    Rajarhat: 290,
    Raumari: 291,
    Ulipur: 292,
  },
  Kushtia: {
    Bheramara: 293,
    Daulatpur: 294,
    Khoksa: 295,
    Kumarkhali: 296,
    "Kushtia Sadar": 297,
    Mirpur: 298,
  },
  Lakshmipur: {
    Kamalnagar: 299,
    "Lakshmipur Sadar": 300,
    Ramganj: 301,
    Ramgati: 302,
    Roypur: 303,
  },
  Lalmonirhat: {
    Aditmari: 304,
    Hatibandha: 305,
    Kaliganj: 306,
    "Lalmonirhat Sadar": 307,
    Patgram: 308,
  },
  Madaripur: {
    Kalkini: 309,
    "Madaripur Sadar": 310,
    Rajoir: 311,
    "Shib Char": 312,
  },
  Magura: {
    "Magura Sadar": 313,
    Mohammadpur: 314,
    Shalikha: 315,
    Sreepur: 316,
  },
  Manikganj: {
    Daulatpur: 317,
    Ghior: 318,
    Harirampur: 319,
    "Manikganj Sadar": 320,
    Saturia: 321,
    Shibalaya: 322,
    Singair: 323,
  },
  Maulvibazar: {
    Barlekha: 324,
    Juri: 325,
    Kamalganj: 326,
    Kulaura: 327,
    "Maulvi Bazar Sadar": 328,
    Rajnagar: 329,
    Sreemangal: 330,
  },
  Meherpur: { Gangni: 331, "Meherpur Sadar": 332, "Mujib Nagar": 333 },
  Munshiganj: {
    Gazaria: 334,
    Lohajang: 335,
    "Munshiganj Sadar": 336,
    Serajdikhan: 337,
    Sreenagar: 338,
    Tongibari: 339,
  },
  Mymensingh: {
    Bhaluka: 340,
    Dhobaura: 341,
    Fulbaria: 342,
    Gaffargaon: 343,
    Gauripur: 344,
    Haluaghat: 345,
    Ishwarganj: 346,
    Muktagachha: 347,
    "Mymensingh Sadar": 348,
    Nandail: 349,
    Phulpur: 350,
    Trishal: 351,
  },
  Naogaon: {
    Atrai: 352,
    Badalgachhi: 353,
    Dhamoirhat: 354,
    Mahadebpur: 355,
    Manda: 356,
    "Naogaon Sadar": 357,
    Niamatpur: 358,
    Patnitala: 359,
    Porsha: 360,
    Raninagar: 361,
    Sapahar: 362,
  },
  Narail: { Kalia: 363, Lohagara: 364, "Narail Sadar": 365 },
  Narayanganj: {
    Araihazar: 366,
    Bandar: 367,
    "Narayanganj Sadar": 368,
    Rupganj: 369,
    Sonargaon: 370,
  },
  Narsingdi: {
    Belabo: 371,
    Manohardi: 372,
    "Narsingdi Sadar": 373,
    Palash: 374,
    Roypura: 375,
    Shibpur: 376,
  },
  Natore: {
    "Bagati Para": 377,
    Baraigram: 378,
    Gurudaspur: 379,
    Lalpur: 380,
    "Natore Sadar": 381,
    Singra: 382,
  },
  Nawabganj: {
    Bholahat: 383,
    Gomastapur: 384,
    Nachole: 385,
    "Nawabganj Sadar": 386,
    Shibganj: 387,
  },
  Netrakona: {
    Atpara: 388,
    Barhatta: 389,
    Durgapur: 390,
    Kalmakanda: 391,
    Kendua: 392,
    Khaliajuri: 393,
    Madan: 394,
    Mohanganj: 395,
    "Netrokona Sadar": 396,
    Purbadhala: 397,
  },
  Nilphamari: {
    Dimla: 398,
    Domar: 399,
    Jaldhaka: 400,
    Kishoreganj: 401,
    "Nilphamari Sadar": 402,
    Saidpur: 403,
  },
  Noakhali: {
    Begumganj: 404,
    Chatkhil: 405,
    Companiganj: 406,
    Hatiya: 407,
    Kabirhat: 408,
    "Noakhali Sadar (Sudharam)": 409,
    Senbagh: 410,
    Sonaimuri: 411,
    Subarnachar: 412,
  },
  Pabna: {
    Atgharia: 413,
    Bera: 414,
    Bhangura: 415,
    Chatmohar: 416,
    Faridpur: 417,
    Ishwardi: 418,
    "Pabna Sadar": 419,
    Santhia: 420,
    Sujanagar: 421,
  },
  Panchagarh: {
    Atwari: 422,
    Boda: 423,
    Debiganj: 424,
    "Panchagarh Sadar": 425,
    Tentulia: 426,
  },
  Patuakhali: {
    Bauphal: 427,
    Dashmina: 428,
    Dumki: 429,
    Galachipa: 430,
    "Kala Para": 431,
    Mirzaganj: 432,
    "Patuakhali Sadar": 433,
  },
  Pirojpur: {
    Bhandaria: 434,
    Kawkhali: 435,
    Mathbaria: 436,
    Nazirpur: 437,
    "Nesarabad (Swarupkati)": 438,
    "Pirojpur Sadar": 439,
    Zianagar: 440,
  },
  Rajbari: {
    "Balia Kandi": 441,
    Goalandaghat: 442,
    Kalukhali: 443,
    Pangsha: 444,
    "Rajbari Sadar": 445,
  },
  Rajshahi: {
    Bagha: 446,
    Baghmara: 447,
    Boalia: 448,
    Charghat: 449,
    Durgapur: 450,
    Godagari: 451,
    Matihar: 452,
    Mohanpur: 453,
    Paba: 454,
    Puthia: 455,
    Rajpara: 456,
    "Shah Makhdum": 457,
    Tanore: 458,
  },
  Rangamati: {
    "Baghai Chhari": 459,
    Barkal: 460,
    "Belai Chhari": 461,
    "Jurai Chhari": 462,
    Kaptai: 463,
    "Kawkhali (Betbunia)": 464,
    Langadu: 465,
    Naniarchar: 466,
    Rajasthali: 467,
    "Rangamati Sadar": 468,
  },
  Rangpur: {
    Badarganj: 469,
    Gangachara: 470,
    Kaunia: 471,
    "Mitha Pukur": 472,
    Pirgachha: 473,
    Pirganj: 474,
    "Rangpur Sadar": 475,
    Taraganj: 476,
  },
  Satkhira: {
    Assasuni: 477,
    Debhata: 478,
    Kalaroa: 479,
    Kaliganj: 480,
    "Satkhira Sadar": 481,
    Shyamnagar: 482,
    Tala: 483,
  },
  Shariatpur: {
    Bhedarganj: 484,
    Damudya: 485,
    Gosairhat: 486,
    Naria: 487,
    "Shariatpur Sadar": 488,
    Zanjira: 489,
  },
  Sherpur: {
    Jhenaigati: 490,
    Nakla: 491,
    Nalitabari: 492,
    "Sherpur Sadar": 493,
    Sreebardi: 494,
  },
  Sirajganj: {
    Belkuchi: 495,
    Chauhali: 496,
    Kamarkhanda: 497,
    Kazipur: 498,
    Royganj: 499,
    Shahjadpur: 500,
    "Sirajganj Sadar": 501,
    Tarash: 502,
    "Ullah Para": 503,
  },
  Sunamganj: {
    Bishwambarpur: 504,
    Chhatak: 505,
    "Dakshin Sunamganj": 506,
    Derai: 507,
    Dharampasha: 508,
    Dowarabazar: 509,
    Jagannathpur: 510,
    Jamalganj: 511,
    Sulla: 512,
    "Sunamganj Sadar": 513,
    Tahirpur: 514,
  },
  Sylhet: {
    Balaganj: 515,
    "Beani Bazar": 516,
    Bishwanath: 517,
    Companiganj: 518,
    "Dakshin Surma": 519,
    Fenchuganj: 520,
    Golabganj: 521,
    Gowainghat: 522,
    Jaintiapur: 523,
    Kanaighat: 524,
    "Sylhet Sadar": 525,
    Zakiganj: 526,
  },
  Tangail: {
    Basail: 527,
    Bhuapur: 528,
    Delduar: 529,
    Dhanbari: 530,
    Ghatail: 531,
    Gopalpur: 532,
    Kalihati: 533,
    Madhupur: 534,
    Mirzapur: 535,
    Nagarpur: 536,
    Sakhipur: 537,
    "Tangail Sadar": 538,
  },
  Thakurgaon: {
    Baliadangi: 539,
    Haripur: 540,
    Pirganj: 541,
    Ranisankail: 542,
    "Thakurgaon Sadar": 543,
  },
};
