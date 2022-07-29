# These aliases were identified as problematic when trying to merge the
# Employment aliases with the Updated Degree aliases
# they were problematic in the sense that they merged multiple employing
# Institution
#
# What I did was:
# - take the updated records aliases and propose merges with the employment aliases
# - I examined any that merged multiple employment agencies
# - I then googled the set and hardcoded the real ones.
EMPLOYMENT_MERGING_ALIASES = [
    {
        "Imperial College London",
        "Imperial College of Science and Technology, London",
    },
    {
        "University of Georgia",
        "Institute of Higher Education, University of Georgia, Athens, Georgia",
    },
    {
        "Jonkoping International Business School",
        "Jonkoping International Business School, Sweden",
    },
    {
        "KAIST",
        "Korea Advanced Institute of Science and Technology (KAIST)",
        "Korea Advanced Institute of Science and Technology",
        "the Korea Advanced Institute of Science and Technology",
    },
    {
        # ^ they are different
        "Korea Institute of Science and Technology",
    },
    {
        "Leningrad State University",
        "Leningrad University",
    },
    {
        "QUEENS UNIVERSITY",
        "Queen's University",
        "Queen's University, Canada",
    },
    {
        "Rutgers - Camden",
        "Rutgers University-Camden",
    },
    {
        "St Louis University",
        "St. Louis University",
    },
    {
        "State University of New York",
        "State University of New York System",
    },
    {
        "Texas A&M University",
        "Texas A&M University System",
    },
    {
        "Texas Tech University",
        "Texas Tech University System",
    },
    {
        "University of Arkansas",
        "University of Arkansas System",
    },
    {
        "University of Bradford",
        "University of Bradford, United Kingdom",
    },
    {
        "University of London St. George's Medical School",
        "University of London St. George's Medical School, United Kingdom",
    },
    {
        "University of Maryland Baltimore, The",
    },
    {
        "University of Maryland, Baltimore County",
    },
    {
        "University of Tennessee, The",
        "University of Tennessee System",
    },
    {
        "University of Texas Health Science Center at San Antonio, The",
        "University of Texas Medical School at San Antonio",
    },
    {
        "University of Wurzburg",
        "University of Wurzburg, Germany",
    },
    {
        "Bernard Revel Graduate School",
        "Yeshiva University",
    },
    {
        "York University",
        "York University, Canada",
    },
]

# Of the merged aliases, this set contains (or has split) all of those that are
# of length three or greater.
LONG_ALIASES = [
    {
        "Ca'Foscari University of Venice",
        'University of Venice, Italy',
        "Ca'Foscari University of Venice, Italy",
        "University of Venice",
    },
    {
        "Chopin's Academy of Music, Poland",
        "State Academy of Music, Krakow, Poland"
    },
    {
        "Queen's University",
        'QUEENS UNIVERSITY', "Queen's University, Canada",
    },
    {
        "St. George's University",
        "St. George's University School of Medicine",
        'St Georges University', "St George's University School Of Medicine",
    },
    {
        'A. F. Ioffe Institute of Physics and Technology',
        "A.I. Ioffe Physical Technical Institute",
        'Ioffe Institute of Physics and Technology, Leningrad',
        'Ioffe Physico Technical Institute',
    },
    {
        'Academy of Medical Sciences, The (UK)',
    },
    {
        'Academy of Science of USSR, Moscow',
        "Academy of Sciences,  former Soviet Union",
        "Academy of Sciences, Moscow",
        'Academy of Sciences of the USSR',
        'Academy of Sciences of Russia',
        "U.S.S.R. Academy of Sciences",
        'The Russian Academy of Sciences',
        'Soviet Academy of Sciences',
        'Russian Academy of Sciences',
    },
    {
        'Academy of Sciences of the Czech Republic',
        'Academy of Sciences in Prague, Czech',
        'Academy of Sciences in Prague',
    },
    {
        'Agra University',
        'University of Agra',
        'Agra University, India',
    },
    {
        'Ains Chams University, Cairo',
        'Ain Shams University, Egypt',
        'Ain Shams University',
        'Ain Shams University, Cairo, Egypt',
    },
    {
        'All Union Cancer Research Center',
        'All-Union Cancer Research Center, Russia',
        'All-Union Cancer Research Center',
    },
    {
        'Andhra Medical College',
        'Andhra Medical College, India',
        'University Of Health Sciences / Andhra Medical College',
    },
    {
        'Andhra University',
        'Andhra University,india',
        'Andhra University, India',
    },
    {
        'Antioch College',
    },
    {
        'Antioch University New England',
    },
    {
        'Antioch University, Los Angeles',
    },
    {
        'Balseiro Institute, Argentina',
        'Balseiro Institute',
        'Institute Balseiro in Argentina',
    },
    {
        'Bar-Ilan University',
        'Bar Ilan University',
        'Bar-llan University',
    },
    {
        'Bogazici University, Istanbul-Turkey',
        'Bogazici University',
        'Bosphorus University in Istanbul',
    },
    {
        'California Polytechnic State University',
        'California State Polytechnic University',
    },
    {
        'California State Polytechnic University, Pomona',
    },
    {
        'California State University, Fullerton',
        'CALIFORNIA STATE UNIV/FULLERTON',
        'California State Univ/Fullerton',
        'California State Univ, Fullerton',
    },
    {
        'Carleton College',
        'Carleton College, Northfield',
        'CARLETON COLLEGE',
    },
    {
        'Center For Research And Advanced Studies',
    },
    {
        'Center For Research And Advanced Studies, Mexico',
    },
    {
        'Center for Cellular and Molecular Biology',
        'Center for Cellular and Molecular Biology, Hyderabad',
        'Center for Cellular and Molecular Biology, India',
    },
    {
        'Center for Research & Advanced Studies, National Polytechnic Institute',
    },
    {
        'Dow Medical College',
        'Dow Medical College, Pakistan',
        'Dow University Of Health Sciences',
    },
    {
        'Dresden University of Technology',
        'Technical University Of Dresden',
        'University of Technology, Dresden',
        'Technical University Dresden',
    },
    {
        'Ecole des Mines de Paris',
        'Ecole des Mines de Paris, France',
        'Ecole des Mines de Paris, Evry (France)',
        "the Ecole des Mines de Paris",
    },
    {
        'Eotvos Lorand University',
        "ELTE University of Budapest",
        'Eotvos University',
        'L. Eotvos University, Budapest, Hungary',
        'Roland Eotvos University, Budapest',
        "Roland Eotvos University",
        "Lorand Eotvos University",
        "University of Sciences Budapest, Hungary",
        "University of Sciences, Hungary",
    },
    {
        'Erasmus University',
        'Erasmus University Rotterdam',
        'Erasmus University Rotterdam, Netherlands',
    },
    {
        'Free University of Brussels',
        'University Libre De Bruxelles',
        'Free University of Brussels, Belgium',
        'Free University of Brussels',
        'Vrije University, Brussels',
        'University of Brussels',
    },
    {
        'Gdansk Medical University, Poland',
        'Medical University of Gdansk',
        'Gdansk Medical University',
        "Gedanensis Medical Academy, Gdansk, Poland",
        "Medical School of Gdansk",
    },
    {
        'Gothenburg University',
        'University of Gothenburg',
        'University of Gothenburg, Sweden',
        'Sweden University of Gothenburg',
    },
    {
        'Graduate Institute of International Studies Geneva',
        'Graduate Institute Geneva, The',
        'Graduate Institute of International Studies, Geneva',
        "Institute of International Studies in Geneva",
        "Institute of International Studies in Geneva, Switzerland",
    },
    {
        'Hacettepe University, Ankara, Turkey',
        'Hacettepe University',
        'Hacettepe University, Turkey',
    },
    {
        'Hahnemann University',
        'Drexel University',
        'Hahnemann Medical College',
        'Hahnemann Medical College, Philadelphia',
    },
    {
        'Hamburg University of Technology',
        'Technical University Hamburg-Harburg',
        'Technical University Hamburg',
    },
    {
        'Hebrew University of Jerusalem',
        'HEBREW UNIVERSITY OF JERUSALEM',
        'University of Jerusalem',
    },
    {
        'Hungary Academy of Sciences, Budapes',
        'Hungarian Academy of Sciences',
        'Hungarian Academy of Sciences, Budapest',
        "Academy of Sciences in Budapest",
        "Academy of Sciences, Budapest",
    },
    {
        'Institute National Polytechnique Grenoble',
        'Institute National Polytechnique Grenoble, France',
        'Institut National Polytechnique de Grenoble, France',
        "Grenoble Institute of Technology",
        "Polytechnic Institute (Grenoble)",
    },
    {
        'Institute of Cancer Research',
    },
    {
        'Institute of Cancer Research, London',
        'Institute of Cancer Research, London University',
    },
    {
        'Institute of Chemical Physics, Russian Academy of Sciences',
        'Institute of Chemical Physics, Russian Academy of Sciences, Moscow',
        'Institute of Chemical',
    },
    {
        'Institute of Chemical Technology, India',
        'Institute of Chemical Technology (ICT) Mumbai',
        "Mumbai University Institute of Chemical Technology, Mumbai",
        'Institute of Chemical Technology',
    },
    {
        'Johannesburg General Hospital, Johannesburg, South Africa.',
    },
    {
        'Kagawa University Medical School, Japan',
        'Kagawa University Medical School',
        'Kagawa Medical School',
        "Kagawa Medical University",
    },
    {
        "Kanpur Technological Institute",
        "Kanpur Technological Institute, India",
        "Indian Institute of Technology Kanpur",
    },
    {
        'Kanazawa University Graduate School of Medicine, Kanazawa',
        'Kanazawa University, Japan',
        'Kanazawa University',
    },
    {
        'Karnatak University',
        'Karnatak University, India',
        'Karnatak University Dharwar',
    },
    {
        'Lokmanya Tilak Municipal Medical College, India',
        'Lokmanya Tilak Municipal Medical College University',
        'Lokmanya Tilak Municipal Medical College',
    },
    {
        'London School of Hygiene and Tropical Medicine',
        'London School Of Hygiene and Tropical Medicine',
        'London School Of Hygiene and Tropical Medicine, United Kingdom',
    },
    {
        'Marine Hydrophysical Institute',
        'Marine Hydrophysical Institute, Ukraine',
        'Marine Hydrophysical Institute, Sevastopol',
    },
    {
        'Max Planck Institute',
        'Max-Planck Institute, Gottingen, Germany',
    },
    {
        'Memorial University of Newfoundland, Canada',
        'School of Social Work, Memorial University, Newfoundland',
        'Memorial University of Newfoundland',
    },
    {
        'Moscow Lomonosov State University',
        'Lomonosov Moscow State University',
        'Moscow State University',
    },
    {
        'NAChinese University of Hong Kong',
        'Chinese University of Hong Kong',
        'The Chinese University of Hong Kong',
    },
    {
        'National Academy of Science',
    },
    {
        'National Academy of Sciences of Ukraine',
        'National Academy of Sciences, Ukraine',
        'National Academy of Sciences of the Ukraine',
        "Ukraine National Academy of Sciences",
    },
    {
        'National Academy of Sciences',
    },
    {
        'National Academy of Sciences, Kiev',
        'National Academy of Sciences of the USSR, Kiev',
        'National Academy of Sciences of Ukraine, Kiev',
    },
    {
        'National Institute of Immunology',
        'National Institute of Immunology (India)',
        'National Institute of Immunology, India',
    },
    {
        'National Institute of Mental Health and Neuroscience, India',
        'National Institute for Mental Health and Neurosciences',
        'National Institute of Mental Health and Neuroscience',
    },
    {
        'National University of Colombia',
        'Universidad National of Colombia',
        'Universidad Nacional de Colombia',
    },
    {
        'New Paltz, State University of New York',
        'SUNY-New Paltz',
        'State University of New York, New Paltz',
    },
    {
        'Newcastle University',
        'University of Newcastle, Australia',
    },
    {
        'Northwest University, Potchefstroom',
        'Potchefstroom University',
        'North-West University',
    },
    {
        'Nottingham Trent University',
    },
    {
        'Odense University',
        'Odense University, Denmark',
        'Odense Universitet',
    },
    {
        'Panjab University, India',
        'Panjab University',
        'Punjab University',
    },
    {
        'Paris School of Economics',
    },
    {
        'Paul Sabatier University',
        'Universite Paul Sabatier',
    },
    {
        'Pierre and Marie Curie University',
        'University of Paris VI (Pierre and Marie Curie)',
    },
    {
        "Institute Marie Curie, Paris",
        "Curie Institute Paris",
    },
    {
        'Postgraduate Institute of Medical Education and Research, India',
        'Postgraduate Institute of Medical Education and Research, Chandigarh',
        'Postgraduate Institute of Medical Education and Research',
        'Post Graduate Institute of Medical Education and Research',
    },
    {
        'Potsdam, State University of New York',
        'State University of New York, Potsdam',
        'State University of New York at Potsdam',
    },
    {
        'Rijksuniversiteit Groningen',
        'Groningen University, Netherlands',
        'Rijksuniversity Groningen, Netherlands',
        'University of Groningen',
        'Gronigen University',
    },
    {
        'Rostov University',
        'Rostov State University, Russia',
        'Rostov State University',
    },
    {
        'Royal Veterinary and Agricultural University',
        'Royal Veterinary and Agricultural University, Denmark',
        'Royal Veterinary and Agriculture University',
    },
    {
        'Sanjay Gandhi Postgraduate Institute of Medical Sciences, India',
        'Sanjay Gandhi Postgraduate Institute of Medical Sciences',
        'Sanjay Gandhi Post Graduate Institute of Medical Sciences Lucknow',
        'Sanjay Gandhi Post Graduate Institute of Medical Sciences Lucknow, India',
    },
    {
        'Savannah State University',
    },
    {
        'Semmelweis Egyetem',
        'Semmelweis University',
        'Semmelweis Medical University',
    },
    {
        'Sofia University',
        'Institute of Transpersonal Psychology',
        'University of Sofia',
    },
    {
        'Sorbonne University, Paris',
        'University of Paris IV-Sorbonne',
        'La Sorbonne, Paris',
        'University of Paris IV (Sorbonne)',
        "La Sorbonne Paris", "La Sorbonne Paris IV",
    },
    {
        'St. Petersburg Polytechnic Institute, Russia',
        'St. Petersburg Polytechnic Institute',
        'St. Petersburg Polytechnic, Russia',
        "Technical Institute of Lenigrad"
    },
    {
        'State University of New York, Syracuse',
    },
    {
        'Steklov Mathematical Institute',
        'Steklov Institute of Mathematics',
        'Steklov Institute of Mathematics, Russia',
        'Steklov Institute (Moscow)',
    },
    {
        'Stellenbosch University',
        'University of Stellenbosch, South Africa',
        'University of Stellenbosch',
    },
    {
        'Technical University Vienna',
        'Vienna University of Technology',
        'Technical University of Vienna',
        'Technical University of Vienna, Austria',
    },
    {
        'Technical University of Warsaw',
        'Warsaw Technical University',
        'Warsaw University of Technology',
        'Polytechnical University of Warsaw',
    },
    {
        'Technische Universitaet Muenchen',
        'TU Munich',
        'Technical University Munich',
    },
    {
        'The Savannah College of Art And Design',
        'Savannah College of Art And Design',
    },
    {
        'The University of Genoa',
        'Universita Di Genova, Italy',
        'Universita Di Genova',
        'University of Genoa, Italy',
    },
    {
        'The University of Northampton',
        'University of Northampton',
        'University of Northampton, United Kingdom',
    },
    {
        'Trinity College Dublin',
        'Trinity College and Theological Seminary',
        'Trinity College, Ireland',
        'Trinity College',
    },
    {
        'UNIVERSITY OF NEW ENGLAND',
        'University of New England, Australia',
        'University of New England',
    },
    {
        'Umea University, Sweden',
        'University of Umea, Sweden',
        'Umea University',
    },
    {
        "University Central of Barcelona",
    },
    {
        "Barcelona School of Architecture", "Higher Technical School of Architecture, Architecture of Barcelona",
    },
    {
        "ESADE Business School, Barcelona",
    },
    {
        'Universidad Autonoma De Ciudad Juarez Escuela De Medicina',
        'Universidad Autonoma De Ciudad Juarez Escuela De Medicina, Mexico',
        'Universidad Autonoma De Ciudad Juarez',
    },
    {
        'Universidad Central de Venezuela, Caracas',
        'Universidad Central de Venezuela',
        'Central University of Venezuela',
    },
    {
        'Universidad Nacional de La Plata',
        'Universidad Nacional de La Plata, Argentina',
        'National University of La Plata',
    },
    {
        'Universidad de la República',
        'Universidad de la Rep±blica',
        'Universidad de la Repñblica, Uruguay',
        'Universidad de la Republica Oriental del Uruguay',
    },
    {
        'Universita Di Verona',
        'University of Verona',
        'Universita Di Verona, Italy',
    },
    {
        'Universitat Pompeu Fabra',
        'Universitat Pompeu',
    },
    {
        'University College Dublin',
        'University College, Dublin',
        "University College, Dublin, Ireland",
    },
    {
        'University College, Galway, Ireland',
        'University College Galway, Ireland',
        'University College Galway',
    },
    {
        'University Denis Diderot',
        'University of Paris VII (Paris Diderot)',
        'University of Denis Diderot',
        'University Denis Diderot, France',
        'University of Denis Diderot, France',
        'Université Paris Diderot - Paris 7',
        "University of Paris XII",
        'University of Paris VII',
        'University of Paris-Jussieu, Paris',
        'University of Paris Diderot (Paris VII)',
        "University of Denis Diderot- Paris7",
        "Denis Diderot University",
        "Rene Diderot University, Paris, France",
        'Paris Diderot University',
        'Paris Diderot University, France',
        'University Denis Diderot',
    },
    {
        'University Nacional de Cordoba, Argentina',
        'University Nacional de Cordoba',
        'Universidad Nacional de Cordoba, Argentina',
        'Universidad Nacional de Córdoba',
    },
    {
        'University Of Ankara, Turkey',
        'Ankara University, Turkey',
        'University Of Ankara',
        'University of Ankara',
    },
    {
        'University of Alexandria, Egypt',
        'Alexandria University',
        'University of Alexandria',
        "Alexandria School of Medicine",
        "Alexandria School of Medicine, Egypt",
    },
    {
        'University of Alicante',
        'Universidad de Alicante',
        'Universidad de Alicante, Spain',
    },
    {
        'University of Buenos Aires',
        'Universidad de Buenos Aires',
        'Universidad del Buenos Aires',
    },
    {
        'University of Cordoba, Spain',
        'University of Cordoba',
        'University of Cordoba, Argentina',
    },
    {
        'University of Dublin',
    },
    {
        'University of Ghent, Belgium',
        'Ghent University',
        'Universiteit Gent',
    },
    {
        'University of Gottingen',
        'Georg-August-Universität Göttingen',
        'Georg-August-Universität',
    },
    {
        'University of Keele',
        'Keele University',
        'UNIVERSITY OF KEELE',
    },
    {
        'University of Liege',
        'University of Liège, Belgium',
        'University of Liege, Belgium',
    },
    {
        'University of Lisbon, Portugal',
        'University of Lisbon Medical School',
        'University of Lisbon',
    },
    {
        'University of Lodz',
        'University of Lodz (Poland )',
        'University of Lodz, Poland',
    },
    {
        'University of Muenster, Germany',
        'Universitaet Muenster, Germany',
        'University of Muenster',
    },
    {
        'University of Mumbai',
        'University of Mumbai, India',
        'Mumbai University',
    },
    {
        'University of NSW Sydney Australia',
        'University of New South Wales, Sydney',
        'UNIVERSITY OF NEW SOUTH WALES',
        'University of New South Wales',
    },
    {
        'University of Nijmegen',
        'Radboud University Nijmegen',
        'Radboud University, Nijmegen',
    },
    {
        'University of North Texas System',
        'University of North Texas',
        'North Texas State University',
    },
    {
        'University of Ontario Institute of Technology',
    },
    {
        'University of Paris - Campus Unknown',
        'University of Paris',
    },
    {
        'University of Paris III (Nouvelle Sorbonne)',
        "University of la Sorbonne Nouvelle, Paris",
        'La Sorbonne Nouvelle',
        'La Sorbonne Nouvelle, France',
        'Universite de Paris III-Nouvelle Sorbonne',
    },
    {
        'University of Paris V (Paris Descartes)',
        'Rene Descartes University Paris',
        'University Rene Descartes',
        "Paris V University",
    },
    {
        'University of Paris VIII',
        'University of Paris VIII (Vincennes in Saint-Denis)',
    },
    {
        'University of Paris 6, France',
        'University of Paris VI',
        'University of Paris 6',
    },
    {
        'University of Paris XI (France)',
        'University of Paris XI (Sud/South)',
        "University of Orsay, Paris XI, France",
        'University of Paris-Sud 11',
        'Universite de Paris XI',
        "South Paris University, Orsay, France",
        "University of Paris at Orsay",
    },
    {
        'University of Pavia', "Universita' di Pavia, Italy",
        'University of Pavia, Italy',
    },
    {
        'University of Pretoria',
    },
    {
        'University of Punjab',
        'University of Punjab, Pakistan',
        'University of Punjab, Punjab.',
    },
    {
        'University of Rennes-1',
        'University of Rennes-1, France',
        'University of Rennes',
    },
    {
        'University of Rochester',
    },
    {
        'University of Saarland',
        'Universitat des Saarlandes',
        'University of the Saarland',
    },
    {
        'University of Stockholm',
        'Stockholm University',
        'Stockholms Universitet',
    },
    {
        'University of Surrey',
    },
    {
        'University of Sussex',
    },
    {
        'University of Technology of Compiegne, France',
        'University de Technologie de Compiegne',
        'University of Technology of Compiegne',
    },
    {
        "Tuebingen University",
        "Tuebingen University, Germany"
        "Universitt Tbingen",
        "University Tubingen",
        "University of Tubingen",
        "University of Tuebingen",
    },
    {
        'University of West Alabama',
    },
    {
        'University of Western Australia',
        'Australia-U Western Australia - Perth',
    },
    {
        'University of Witwatersrand',
        'Witwatersrand University',
        'University of the Witwatersrand',
    },
    {
        'Vrije Universiteit Amsterdam',
    },
    {
        'Walden University in Minneapolis, MN',
        'Walden University',
    },
    {
        'Warsaw University',
        'University of Warsaw',
    },
    {
        'West Virginia State University',
    },
    {
        'West Virginia University of Technology',
    },
    {
        'West Virginia University',
    },
    {
        'Western Ontario University',
        'University of Western Ontario',
    },
    {
        'the Miguel Hernandez University',
        'Miguel Hernßndez University',
        'Miguel Hernández University, Spain',
    },
]

# this is the set of aliases that match on a significant number of tokens
TOKEN_MATCHED_ALIASES = [
    {
        "Robert Gordon's University",
        'Robert Gordon University',
    },
    {
        "Saint Michaels's College",
        'Saint Michaels College',
    },
    {
        'Aberystwyth',
        'Aberystwyth University',
    },
    {
        'Aix-en-Provence',
        'Aix-en-Provence U.',
    },
    {
        'Alfred Wegener Institute for Polar and Marine Research',
        'Alfred Wegener Institute for Polar and Marine Research, Germany',
    },
    {
        'Architectural Association School of Architecture, London (England)',
        'Architectural Association School of Architecture, London',
    },
    {
        'Augsburg College, Minneapolis',
        'Augsburg University, Minneapolis',
    },
    {
        'Beijing Institue of Pharmacology and Toxicology',
        'Beijing Institute of Pharmacology and Toxicology, China',
    },
    {
        'Belgrade University',
        'University of Belgrade',
    },
    {
        'Bombay University',
        'University of Bombay',
    },
    {
        'Cancer Research Institute, Mumbai',
        'Cancer Research Institute, University of Mumbai, India',
    },
    {
        'Catholic University of Leuven',
        'Catholic University, Leuven',
    },
    {
        'Central Food Technological Research Institute, India',
        'Central Food Technological Research Institute',
    },
    {
        'Charing Cross and Westminster Medical School',
        'Charing Cross and Westminster Medical Schools',
    },
    {
        'Chinese Academy of Medical Sciences, China',
        'Chinese Academy of Medical Sciences',
    },
    {
        'Clarkson College of Technology',
        'Clarkson College',
    },
    {
        'Cologne University',
        'University of Cologne',
    },
    {
        'Cranbrook Academy of Art',
        'Cranbook Academy of Art',
    },
    {
        'Dalian Institute of Chemical Physics, China',
        'Dalian Institute of Chemical Physics',
    },
    {
        'Defence Research and Development Establishment, India',
        'Defence Research and Development Establishment',
    },
    {
        'EUDE Business School - Madrid, Spain',
        'EUDE Business School - Madrid',
    },
    {
        'Embry-Riddle Aeronautical U',
        'Embry-Riddle Aeronautical University',
    },
    {
        'Faculdade de Ciencias Agrarias do Planalto Central, Brazil',
        'Faculdade de Ciencias Agrarias do Planalto Central',
    },
    {
        'Facultes des Sciences de Paris',
        'Facultes des Sciences de Paris, France',
    },
    {
        'Frostburg State U',
        'Frostburg State University',
    },
    {
        'George Fox College',
        'George Fox University',
    },
    {
        'Grace College & Theological Seminary',
        'Grace Theological Seminary',
    },
    {
        'Grenoble University',
        'University of Grenoble',
    },
    {
        'Herriot Watt University',
        'Heriot Watt University',
    },
    {
        'Hull University',
        'University of Hull',
    },
    {
        'Indian Institute of Chemical Biology',
        'Indian Institute of Chemical Biology, India',
    },
    {
        'Indian Statistical Institute',
        'The Indian Statistical Institute',
    },
    {
        'Institut Francais du Petrole (IFP), France',
        'Institut Francais du Petrole (IFP)',
    },
    {
        'Institut National De Recherche En Informatique Et En Automatique',
        'Institut National De Recherche En Informatique Et En Automatique, France',
    },
    {
        'Institute for Pharmazeutishe Biologie, Germany',
        'Institute for Pharmazeutishe Biologie',
    },
    {
        'Institute for Problems of Material Sciences, Kiev',
        'Institute for Problems of Materials Science, Kiev',
    },
    {
        'Institute of Atmospheric Physics, China',
        'Institute of Atmospheric Physics',
    },
    {
        'Institute of Biochemistry',
        'the Institute of Biochemistry',
    },
    {
        'Institute of Bioorganic Chemistry',
        'Institute of Bioorganic Chemistry, Russia',
    },
    {
        'Institute of Genetics and Selection of Industrial Microorganisms, Russia',
        'Institute of Genetics and Selection of Industrial Microorganisms',
    },
    {
        'Iuliu Hatieganu University of Medicine and Pharmacy',
        'Iuliu Hatieganu University of Medicine and Pharmacy, Romania',
    },
    {
        'John Innes Institute, Norwich',
        'John Innes Institute, Norwich, UK',
    },
    {
        'Kuwait Institute of Medical Specialization, Kuwait',
        'Kuwait Institute of Medical Specialization',
    },
    {
        'Landau Institute for Theoretical Physics',
        'Landau Institute for Theoretical Physics, Moscow',
    },
    {
        'Leningrad Electrical Engineering Institute',
        'Leningrad Electrical Engineering Institute, USSR',
    },
    {
        'Loughborough University of Technology',
        'Loughborough University',
    },
    {
        'Loyola University New Orleans College of Law',
        'Loyola University New Orleans',
    },
    {
        'Lucknow University',
        'University of Lucknow',
    },
    {
        'Maastricht University',
        'University of Maastricht',
    },
    {
        'Minneapolis College of Art And Design',
        'Minneapolis College of Art & Design',
    },
    {
        'Montreal University',
        'University of Montreal',
    },
    {
        'Moscow Power Engineering Institute',
        'Moscow Power Engineering Institute (MPEI)',
    },
    {
        'NAChinese University of Hong Kong',
        'Chinese University of Hong Kong',
        'The Chinese University of Hong Kong',
    },
    {
        'National Centre for Cell Science, India',
        'National Centre for Cell Science',
    },
    {
        'National Institute for Pure and Applied Mathematics (IMPA), Brazil',
        'National Institute for Pure and Applied Mathematics (IMPA)',
    },
    {
        'National Institute of Immunology',
        'National Institute of Immunology (India)',
        'National Institute of Immunology, India',
    },
    {
        'National Institute of Public Health, Poland',
        'National Institute of Public Health',
    },
    {
        'National Polytechnic Institute, Mexico',
        'National Polytechnic Institute',
    },
    {
        'National School of Biological Sciences, Mexico',
        'National School of Biological Sciences',
    },
    {
        'National University Federico Villarreal',
        'National University Federico Villarreal, Lima',
    },
    {
        'National Veterinary School of Toulouse, France',
        'National Veterinary School of Toulouse',
    },
    {
        'Nencki Institute of Experimental Biology, Poland',
        'Nencki Institute of Experimental Biology',
    },
    {
        'Northumbria University',
        'University of Northumbria',
    },
    {
        'Novosibirsk Institute of Organic Chemistry',
        'Novosibirsk Institute of Organic Chemistry, Russia',
    },
    {
        'Paris Dauphine University',
        'University of Paris IX Dauphine',
    },
    {
        'Petrov Cancer Research Institutes Petersburg, Russia',
        'Petrov Cancer Research Institutes Petersburg',
    },
    {
        'Plymouth University',
        'University of Plymouth',
    },
    {
        'Pontifical Gregorian University, Rome',
        'Pontifical Gregorian University in Rome',
    },
    {
        'Research Institute for Applied Microbiology, Russia',
        'Research Institute for Applied Microbiology',
    },
    {
        'Rheinische Friedrich Wilhelms Universitat',
        'Rheinische Friedrich Wilhelms University, Bonn',
        'Rheinische Friedrich Wilhelms Universitat, Germany',
    },
    {
        'Saha Institute of Nuclear Physics, India',
        'Saha Institute of Nuclear Physics',
    },
    {
        'School of Veterinary MedicinSchool of Veterinary Medicine Warsaw',
        'School of Veterinary MedicinSchool of Veterinary Medicine Warsaw, Poland',
    },
    {
        'Shaheed Beheshti University of Medical Sciences (Sbums)',
        'Shahid Beheshti University of Medical Sciences (Sbums)',
    },
    {
        'Shahid Beheshti University',
        'Shaheed Beheshti University',
    },
    {
        'Southern Illinois University at Edwardsville',
        'Southern Illinois University, Edwardsville',
    },
    {
        'St. Joseph College', "St. Joseph's College",
    },
    {
        'State University of Music and Performing Arts Stuttgart',
        'State University of Music and Performing Arts Stuttgart, Germany',
    },
    {
        'Sun-Yet Sun University of Medical Science',
        'Sun-Yet Sen University of Medical Sciences',
    },
    {
        'Swansea University',
        'University of Swansea',
    },
    {
        'Technical University of Budapest',
        'Technical University, Budapest',
    },
    {
        'Tehran University',
        'University of Tehran',
    },
    {
        'The Cleveland Institute of Music',
        'Cleveland Institute of Music',
    },
    {
        'The Hong Kong University of Science and Technology',
        'Hong Kong University of Science and Technology',
    },
    {
        'The Union Institute and University',
        'Union Institute and University',
    },
    {
        'The University of Aarhus',
        'Aarhus University',
    },
    {
        'Tomsk State University and St.-Petersburg State University',
        'Tomsk State University and St.-Petersburg State University, Russia',
        'Tomsk State University',
        'Tomsk State University, Russia',
        "Tomsk State University (USSR)",
    },
    {
        "Tomsk Polytechnic Institute",
        "Tomsk Polytechnic University, USSR",
    },
    {
        'Universidad Nacional Mayor de San Marcos',
        'Universidad Nacional Mayor de San Marcos, Peru',
    },
    {
        'Universidad Nacional Pedro Henriquez Ureña',
        'Universidad Nacional Pedro Henriquez Urena',
    },
    {
        'Universidad Nacional de San Agustin, Peru',
        'Universidad Nacional de San Agustin',
    },
    {
        'Universidade Federal de Minas Gerais, Brazil',
        'Universidade Federal de Minas Gerais',
        'Federal University of Minas Gerais, Belo Horizonte',
        'Federal University of Minas Gerais',
    },
    {
        'Universidade Federal do Parana, Brazil',
        'Universidade Federal do Parana',
    },
    {
        'Universidade do Estado do Rio de Janeiro, Brazil',
        'Universidade do Estado do Rio de Janeiro',
        'Federal University of Rio de Janeiro, Brazil',
        'Federal University of Rio de Janeiro',
    },
    {
        'Universita degli Studi di Perugia, Italy',
        'Universita degli Studi di Perugia',
    },
    {
        'Universitatea de Medicina Si Farmacie Victor Babes, Romania',
        'Universitatea de Medicina Si Farmacie Victor Babes',
    },
    {
        'Universite De Toulouse-Le Mirail, France',
        'Universite De Toulouse-Le Mirail',
    },
    {
        'Universite Jean Monnet Saint-Etienne, France',
        'Universite Jean Monnet Saint-Etienne',
        "University of Saint-Etienne",
    },
    {
        'University La Sapienza of Rome',
        'University La Sapienza, Rome',
    },
    {
        'University Of Antioquia, Medellin, Columbia',
        'University of Antioquia, Medellin, Colombia',
    },
    {
        'University of Auckland',
        'Auckland University',
    },
    {
        'University of Basel',
        'Basel University',
        "University of Basle",
    },
    {
        'University of Cairo',
        'Cairo University',
    },
    {
        'University of Delhi',
        'Delhi University',
    },
    {
        'University of Duesseldorf',
        'Duesseldorf University',
    },
    {
        'University of Lancaster',
        'Lancaster University',
    },
    {
        'University of Pikeville',
        'Pikeville College',
    },
    {
        'University of Salford',
        'Salford University',
    },
    {
        'University of Southern Denmark',
        'University Southern Denmark',
    },
    {
        'University of St Andrews', "St. Andrew's University",
    },
    {
        'University of Ulster',
        'Ulster University',
    },
    {
        'University of Uppsala',
        'Uppsala University',
    },
    {
        'University of Waikato',
        'Waikato University',
    },
    {
        'University of the Sciences in Philadelphia',
        'University of Sciences in Philadelphia',
    },
    {
        'Vienna University of Economics and Business Administration, Austria',
        'Vienna University of Economics and Business Administration',
    },
    {
        'Wageningen University',
        'University of Wageningen',
    },
    {
        'Washington & Lee University',
        'Washington and Lee University',
    },
    {
        'Waynesburg University',
        'Waynesburg College',
    },
    {
        'the University of Bochum',
        'University of Bochum',
    },
]

# After all other steps, the only remaining aliases are of length two.
# I have googled and split all of these.
TWO_LENGTH_ALIASES = [
    {
        "King George's Medical College",
        'King George’s Medical College, India',
    },
    {
        "People's Friendship University, Moscow",
        'Peoples Friendship University of Russia, Moscow',
    },
    {
        "St. Joseph's University, Lebanon",
        'St. Joseph University',
    },
    {
        'AGH University of Science and Technology, Krakow, Poland',
        'AGH University of Science and Technology',
        "Technical University AGH, Krakow",
    },
    {
        'Aachen Technical University',
        'Technical University Aachen, Germany,',
    },
    {
        'Aalborg University, Denmark',
        'Aalborg University',
    },
    {
        'Aalto University, Finland',
        'Aalto University',
    },
    {
        'Aarhus School of Architecture, Denmark',
        'Aarhus School of Architecture',
    },
    {
        'Abo Akademi University, Finland',
        'Abo Akademi University',
    },
    {
        'Academia Alfonsiana, Rome, Italy',
        'Academia Alphonsianum, Rome',
    },
    {
        'Academy of Fine Arts',
    },
    {
        'Academy of Fine Arts and Design, Wroclaw, Poland',
    },
    {
        'Academy of Medical Sciences Moscow',
        'Academy Medical Science of USSR-Russia',
        "Academy of Medical Sciences of the USSR",
    },
    {
        'Aleppo School of Medicine, Syria',
        'Aleppo',
    },
    {
        'Aristotle University of Thessaloniki, Greece',
        'Aristotle University of Thessaloniki',
    },
    {
        'Armed Forces Medical College, India',
        'Armed Forces Medical College',
    },
    {
        'Asbury College, Wilmore',
        'Asbury College',
    },
    {
        'Asian Institute of Technology, Thailand',
        'Asian Institute of Technology',
    },
    {
        'Athens University Economics and Business',
        'Athens University Economics and Business, Greece',
    },
    {
        'Austin Peay State University, Clarksville',
        'Austin Peay State University',
    },
    {
        'Autonoma University of Madrid',
        'Autonoma University',
    },
    {
        'Babes-Bolyai University',
        'Babes-Bolyai University, Romania',
    },
    {
        'Bangalore University, India',
        'Bangalore University',
    },
    {
        'Bariloche University, Argentina',
        'Bariloche University',
    },
    {
        'Baroda University',
        'Baroda University, India',
        'University of Baroda',
        'University of Baroda, India',
    },
    {
        'Belarusian State University',
        'Belorussian State University, Minsk',
        'Belarus State University, Minsk, Belarus',
        'Belarussian State Univesity',
        "Belorussian State University",
        "University of Belorussia",
        "Byelorussian State University",
    },
    {
        'Bengbu Medical College',
        'Bengbu Medical College Anhui, China',
    },
    {
        'Bharathiar University, India',
        'Bharathiar University',
    },
    {
        'Bielefeld University, Germany',
        'Bielefeld University',
    },
    {
        'Bilkent University, Turkey',
        'Bilkent University',
    },
    {
        'Birla Institute of Technology and Science',
        'Birla Institute of Technology',
    },
    {
        'Bocconi University',
        'Bocconi University, Milan',
    },
    {
        'Brandenburg Technical University, Germany',
        'Brandenburg Technical University',
    },
    {
        'Brighton University England',
        'University of Brighton',
    },
    {
        'Brock University, Canada',
        'Brock University',
    },
    {
        'California Polytechnic State University, San Luis Obispo',
        'CAL STATE POLYTECH S L OBISPO',
    },
    {
        'Campinas State University',
        'Campinas State University, Brazil',
    },
    {
        'Capital University',
        'Capital University Law School',
    },
    {
        'Catholic University, Santiago, Chile',
        'Catholic University of Chile',
    },
    {
        'Cayetano Heredia University',
        'Cayetano Heredia University, Lima',
    },
    {
        'Central Missouri State University',
        'University of Central Missouri',
    },
    {
        'Chicago School of Professional Psychology, LA Campus',
        'Chicago School of Professional Psychology',
    },
    {
        'China Agricultural University',
        "Agricultural University Beijing",
        'China Agricultural University, Beijing, China',
        "China (Beijing) Agricultural University",
        'China Agricultural University, Beijing, China',
        'Beijing Agricultural University',
    },
    {
        'Christian Medical College, India',
        'Christian Medical College',
    },
    {
        'City University of Hong Kong, Hong Kong',
        'City University of Hong Kong',
    },
    {
        'City University, London',
        'City University in London',
    },
    {
        'Claude-Bernard University',
        'Lyon 1 University',
    },
    {
        'Clausthal University of Technology, Germany',
        'Clausthal University of Technology',
    },
    {
        'Cordoba National University',
        'National University of Cordoba',
    },
    {
        'Courtauld Institute of Art',
        'Courtauld Institute of Art, London',
    },
    {
        'Cracow University of Technology, Poland',
        'Cracow University of Technology',
        "Krakow University of Technology",
        "Technical University of Krakow",
        "University of Science and Technology, Krakow",
    },
    {
        'Czech Technical University Of Prague',
        'Czech Technical University Of Prague, Czechoslovakia',
        "Czech Technical University",
        "Czech Technical University in Prague",
    },
    {
        'Dagestan Medical Academy, Russia',
        'Dagestan Medical Academy',
    },
    {
        'Darmstadt University of Technology',
        "University of Technology, Darmstadt",
        'Technical University Darmstadt',
    },
    {
        'Dayanand Medical College',
        'Dayanand Medical College, India',
    },
    {
        'Deakin University, Australia',
        'Deakin University',
    },
    {
        'Delft University of Technology, The Netherlands',
        'Delft University of Technology',
        'University of Delft, Netherlands',
        "Delft University",
    },
    {
        'Devi Ahilya University',
    },
    {
        'Banaras Hindu University',
    },
    {
        'Dnepropetrovsk University',
        'Dnepropetrovsk University, Ukraine',
        'Dnipropetrovsk National University, Ukraine',
        'Dnipropetrovsk National University',
    },
    {
        'Dominican University',
        'Dominican University of California',
    },
    {
        'Dublin City University, Ireland',
        'Dublin City University',
    },
    {
        'East-West Center',
        'East-West Center/University',
    },
    {
        'Ecole Centrale de Lille, France',
        'Ecole Centrale de Lille',
    },
    {
        'Ecole Centrale of Paris, France',
        'Ecole Centrale of Paris',
    },
    {
        'Edinburgh College of Art, Edinburgh, Scotland',
        'University of Edinburgh, The',
    },
    {
        'Ehess And Paris School Of Economics',
        'EHESS & PSE-DELTA (France)',
    },
    {
        'Eidgenossische Technische, Zurich',
        'ETH Zurich',
        'Federal Institute of Technology Switzerland',
        'Federal Institute of Technology, Zurich, Switzerland',
    },
    {
        'Eindhoven University of Technology',
        'Eindhoven University of Technology, Netherlands',
        "Eindhoven Technical University",
    },
    {
        'Escola Paulista de Medicina-EPM (currently UNIFESP), Sao Paulo, Brazil',
        'Escola Paulista de Medicina',
    },
    {
        'Estonian Academy of Sciences, Tartu',
        'Estonian Academy of Sciences',
    },
    {
        'European Graduate School, Switzerland',
        'European Graduate School',
    },
    {
        'Far Eastern State University',
        'Far Eastern University',
    },
    {
        'Federal University of Campina Grande',
        'Federal University of Campina Grande, Brazil',
    },
    {
        'Federal University of Ceara',
        'Federal University of Ceara, Brazil',
    },
    {
        'Federal University of Mato Grosso do Sul, Brazil',
        'Federal University of Mato Grosso do Sul',
    },
    {
        'Federal University of Rio Grande do Sul, Brazil',
        'Federal University of Rio Grande do Sul',
        'Federal University of Rio Grande do Sul',
        'Universidade Federal do Rio Grande do Sul',
        "University of Rio Grande Do Sul (Brazil)",
    },
    {
        'Federal University of Santa Catarina, Brazil',
        'Federal University of Santa Catarina',
    },
    {
        'Federal University of Santa Maria',
        'Federal University of Santa Maria, Brazil',
    },
    {
        'Finch University of Health Sciences',
        'Finch University of Health Sciences, Chicago',
    },
    {
        'First Military Medical College - China',
        'First Military Medical University',
    },
    {
        'Friedrich-Alexander-University, Erlangen, Germany',
        'Friedrich-Alexander Universitat',
        'Friedrich-Schiller University of Jena',
        'Friedrich-Schiller University of Jena, Germany',
    },
    {
        'Gabrichevsky Research Institute of Epidemiology and Microbiology, Moscow',
        'Gabrichevsky Institute',
    },
    {
        'German Sport University',
        'German Sport University, Germany',
    },
    {
        'Glasgow Caledonian University, United Kingdom',
        'Glasgow Caledonian University',
    },
    {
        'Gorakhpur University, India',
        'Gorakhpur University',
    },
    {
        'Gorky State University',
        'Gorky State University (Russia)',
    },
    {
        'Goteborg University',
        'University of Goteborg, Goteborg, Sweden',
    },
    {
        'Government Medical College, India',
        'Government Medical College',
    },
    {
        'Graz University of Technology',
        'Graz University of Technology, Austria',
    },
    {
        'Gregorian University',
        'Gregorian University, Rome',
    },
    {
        'Griffith University',
        'Griffith University, Australia',
    },
    {
        'Gulbarga University',
        'Government Medical College, Gulbarga University, India',
    },
    {
        'Guru Nanak Dev University, India',
        'Guru Nanak Dev University',
    },
    {
        'HEC',
        'HEC Paris',
    },
    {
        'Hartford Seminary Foundation',
        'Hartford Seminary College, Hartford',
    },
    {
        'Herzen State Pedagogical University',
        'Herzen State Pedagogical University, Russia',
        "St. Petersburg Hertzen",
        "St. Petersburg State Pedagogical University",
        "Leningrad Pedagogical Institute, USSR",
    },
    {
        'Himachal Pradesh University, India',
        'Himachal Pradesh University',
    },
    {
        'Hirszfeld Institute of Immunology',
        'Ludwik Hirszfeld Institute of Immunology and Experimental Therapy',
    },
    {
        'Hochschule Anhalt, Germany',
        'Hochschule Anhalt',
    },
    {
        'Hohenheim University, Germany',
        'Hohenheim University',
    },
    {
        'IIT Delhi, India',
        'IIT Delhi',
    },
    {
        'Iasi University, Romania',
        'Iasi University',
    },
    {
        'Imperial College',
        'Imperial College, Brazil',
    },
    {
        'Indian Veterinary research Institute',
        'Indian Veterinary Research Institute, Izatnagar, India',
    },
    {
        'Inha University',
        'Inha University, Korea',
    },
    {
        'Institut Catholique, Paris, France',
        'Institut Catholique de Paris',
    },
    {
        'Institut National des Sciences',
        'Institut National des Sciences Appliquées (INSA)',
    },
    {
        "Academy of Sciences of Uzbekistan",
    },
    {
        'Institute For Physics And Biophysics, Uzbekistan',
        'Institute For Physics And Biophysics',
        "Tashkent Institute of Physiology and Biophysics, Uzbekistan",
    },
    {
        'Institute Of Nuclear Research, Poland',
        'Institute Of Nuclear Research',
    },
    {
        'Institute for Genetik',
        'Institute for Genetik, Germany',
    },
    {
        'Institute for Nuclear Research and Nuclear Energy',
        'Institute For Nuclear Research And Nuclear Energy, Sofia, Bulgaria',
    },
    {
        'Institute of Applied Physics of the Russian Academy of Sciences (Russia)',
        'Research Institute of Applied Physics, Moscow',
    },
    {
        'Institute of Cell Biophysics, Russia',
        'Institute of Cell Biophysics',
    },
    {
        'Institute of Cytology of Russian Academy of Sciences',
        "Institute of Cytology, Leningrad St Petersburg",
        'Institute Of Cytology Academy Of Sciences, Leningrad, Russia',
        'Institute of Cytology Russian Academy of Sciences, St. Petersburg',
        "Institute Cytology, St. Petersburg, Russia",
        "Institute of Cytology Academy of Sciences"
    },
    {
        'Institute of Genetics, Russia',
        'Institute of Genetics',
    },
    {
        'Institute of Mathematics',
        'Institute of Mathematics, Romania',
    },
    {
        'Institute of Medicine(1) Yangon, Myanmar ',
        'Institute of Medicine(1) Yangon',
    },
    {
        'Institute of Molecular Biology, Moscow',
        'Institute of Molecular Biology Russian Academy of Sciences, Moscow',
    },
    {
        'Institute of Physiology, Bulgaria',
        'Institute of Physiology',
    },
    {
        'Institute of Protein Research Russian Academy of Sciences',
        'Institute of Protein Research & Moscow State University, Russia',
    },
    {
        'Institute of Psychiatry, University of London',
    },
    {
        "King's College London",
    },
    {
        'Instituto Balseiro',
        'Instituto Balseiro, Argentina',
    },
    {
        'Interamerican University of Puerto Rico - San German',
        'Interamerican University of Puerto Rico',
    },
    {
        'International Academy of Philosophy',
        'International Academy of Philosophy, Liechtenstein',
    },
    {
        'International School of Modena',
        'International School of Modena, Italy',
    },
    {
        'Istanbul University, Turkey',
        'Istanbul University',
    },
    {
        'Javeriana University in Bogota, Columbia',
        'Javeriana University',
    },
    {
        'Jordan University of Science And Technology, Jordan',
        'Jordan University of Science And Technology',
    },
    {
        'Julius-Maximilians-Universität Würzburg',
        'Bayerische Julius-Maximilians-Universitat Wurzburg',
    },
    {
        'Kabul Medical University',
        'Kabul Medical University, Afghanistan',
    },
    {
        'Kagoshima University, Japan',
        'Kagoshima University',
        "University of Kagoshima",
        "Kagoshima University in Japan"
    },
    {
        'Kanazawa College of Art',
        'Kanazawa College of Art, Japan',
    },
    {
        'Kaohsiung Medical College, Kaohsiung, Taiwan',
        'Kaohsiung Medical University',
    },
    {
        'Kapitza Institute, Moscow',
        'Kapitza Institute for Physical Problems, Moscow',
    },
    {
        'Karolinska Institute in Stockholm',
        'Karolinska Institute',
    },
    {
        'Kasturba Medical College, Manipal University, India',
        'Kasturba Medical College',
    },
    {
        'Katholieke Universiteit Leuven',
        'Katholieke Universiteit Leuven, Belgium',
    },
    {
        'Kaunas Medical Academy',
        'Kaunas Medical Academy, Lithuania',
    },
    {
        'Kenyatta Universityirobi, Kenya',
        'Kenyatta Universityirobi',
    },
    {
        'Kharkov National University, Ukraine',
        'Kharkov National University',
        'Kharkov University, Ukraine',
        'Kharkov University',
    },
    {
        'Kharkov Polytechnic University, Ukraine, Russia',
        'Kharkov Polytechnic Institute',
    },
    {
        'King Edward Medical College, Lahore',
        'King Edward Medical College',
    },
    {
        'Kochi Medical Graduate School',
        'Kochi Medical School',
        "Kochi Medical School, Japan",
    },
    {
        'Kossuth Lajos University',
        'Kossuth Lajos University, Hungary',
        "Lajos Kossuth University",
        "Kossuth University, Debrecen, Hungary",
    },
    {
        'Kumamoto University, Japan',
        'Kumamoto University',
    },
    {
        'Kumaun University, Nainital, India',
        'Kumaun University',
    },
    {
        'Kurnool Medical College',
        'Kurnool Med Coll, Univ Hlth Sci, Kurnoo',
    },
    {
        'Lake Erie College of Osteopathic Medicine, Erie Campus',
        'Lake Erie College of Osteopathic Medicine',
    },
    {
        'Lake University, San Antonio',
        'Lake University',
    },
    {
        'Latvia State University',
        'University of Latvia',
    },
    {
        'Lehman College (CUNY)',
        'Lehman College',
    },
    {
        'Leningrad Polytechnic',
        'Leningrad Polytechnic Institute',
    },
    {
        'Leyden University, Netherlands',
        'Leyden University',
    },
    {
        'Lincoln University, New Zealand',
        'Lincoln University',
    },
    {
        'Linkoping University',
        'Linkoping University, Sweden',
    },
    {
        'Liverpool John Moores University, United Kingdom',
        'Liverpool John Moores University',
    },
    {
        "Liverpool School",
        "Liverpool School of Tropical Medicine, Liverpool UK",
    },
    {
        'Lodz University of Technology, Poland',
        'Lodz University of Technology',
        'Technical University of Lodz, Poland',
        'Technical University of Lodz',
    },
    {
        'Louisiana Tech University',
    },
    {
        'Northwestern State University',
    },
    {
        'Ludwig Maximilian University of Munich',
        'University of Munich',
    },
    {
        'Lund University',
        'University of Lund, Sweden',
    },
    {
        'Lviv School of Medicine - Lviv, Ukraine',
        'Lviv School of Medicine - Lviv',
        'Lviv State University',
        'Medical University of Lviv',
    },
    {
        'MRC Human Genetics Unit, Scotland',
        'MRC Human Genetics Unit',
    },
    {
        'Macquarie University',
        'Macquarie University, Australia',
    },
    {
        'Madras Medical College, India',
        'Madras Medical College',
    },
    {
        'Maharaja Sayajirao University of Baroda',
        'Maharaja Sayajirao University of Baroda, India',
    },
    {
        'Mahatma Gandhi University',
        'Mahatma Gandhi University, India',
    },
    {
        'Manchester Business School, United Kingdom',
        'Manchester Business School',
    },
    {
        'Medical Academy, Poland',
        'Medical Academy',
    },
    {
        'Medical Institute in Kiev, Ukraine',
        'Medical Institute in Kiev',
        "Kiev Medical University",
        "Kiev Medical University, Ukraine",
        "Kiev National Medical University, Ukraine"
    },
    {
        'Medical School Of Hannover, Germany',
        'Medical School Of Hannover',
    },
    {
        'Medical School in Wroclaw',
        'Medical School in Wroclaw, Poland',
    },
    {
        'Medical University Of Pecs, Hungary',
        'Medical University Of Pecs',
    },
    {
        'Medical University of Lublin',
        'Medical University of Lublin, Poland',
        "Akademia Medyczna, Lublin",
    },
    {
        'Medical University of Sofia',
        'Medical University Sofia Bulgaria',
    },
    {
        'Medical University of Vienna, Vienna',
        'Medical University of Vienna',
    },
    {
        'Memphis State University',
        'University of Memphis',
    },
    {
        'Midwestern University',
        'Chicago College of Osteopathic Medicine',
    },
    {
        'Mills College',
        'Mills College, Oakland',
    },
    {
        'Mississippi College',
        'Mississippi College, Clinton',
    },
    {
        "Mississippi Medical Center",
    },
    {
        "Mississippi University for Women",
    },
    {
        'Moldova State University, Moldova',
        'Moldova State University',
    },
    {
        "Kitami Institute of Technology (KIT)",
        "Kitami Institute of Technology (KIT), Japan",
    },
    {
        'Moscow Aviation Institute, Russia',
        'Moscow Aviation Institute',
    },
    {
        'Moscow Conservatory',
        'Moscow State Conservatory, Moscow, USSR',
    },
    {
        'Moscow Institute of Physics and Technology',
    },
    {
        'Institute of Control Sciences, Moscow',
    },
    {
        'Musikhochschule Freiburg, Germany',
        'Musikhochschule Freiburg',
    },
    {
        'Mysore Medical College',
        'Government Medical College, Mysore University, India',
    },
    {
        'Nagasaki University',
        'Nagaski University, Nagaski',
    },
    {
        'Nagpur University, India',
        'Nagpur University',
    },
    {
        'Nanjing Medical University',
        'Nanjing Medical University, Nanjing ,China',
    },
    {
        'Nanyang Technological University, Singapore',
        'Nanyang Technological University',
    },
    {
        'National Academy of Theatre and Film Arts, Bulgaria',
        'National Academy of Theatre and Film Arts',
    },
    {
        'National Cancer Institute, Bethesda',
        'United States National Cancer Institute (NIH)',
    },
    {
        'National Pingtung University of Science and Technology, Taiwan',
        'National Pingtung University of Science and Technology',
    },
    {
        'National Theatre Conservatory',
        'National Theatre Conservatory, Denver Center for the Performing Arts',
    },
    {
        'National University At Rosario',
        'National University At Rosario, Argentina',
    },
    {
        'National University of Asuncion',
        'National University of Asuncion, Paraguay',
    },
    {
        'National University of Athens, Greece',
        'National University of Athens',
    },
    {
        'National University of Madrid, Spain',
        'National University of Madrid',
    },
    {
        'Nationale Autonoma Universidad de Mexico',
        'National University of Mexico, Mexico City',
    },
    {
        'New England Conservatory of Music',
        'New England Conservatory',
    },
    {
        'New England School of Law, Boston',
        'New England School of Law',
    },
    {
        'New School for Social Research',
        'New School, The',
    },
    {
        'North Eastern Hill University, India',
        'North Eastern Hill University',
    },
    {
        'Norwegian University of Science and Technology',
        'Norwegian University of Science',
    },
    {
        'Novosibirsk Institute of Bioorganic Chemistry, Russia',
        'Novosibirsk Institute of Bioorganic',
    },
    {
        'Novosibirsk University',
        'Novosibirsk State University',
    },
    {
        'Obafemi Awolowo University',
        'Obafemi Awolowo University, Nigeria',
    },
    {
        'Oral Roberts University School of Medicine, Tulsa',
        'Oral Roberts University',
    },
    {
        'Oregon Graduate Institute of Science and Technology',
        'Oregon Graduate Institute',
    },
    {
        'Oswego, State University of New York',
        'Oswego State University',
    },
    {
        'Pantnagar University, India',
        'Pantnagar University',
    },
    {
        'Pasteur Institute',
        'Institut Pasteur',
    },
    {
        'Philadelphia College of Osteopathic Medicine',
        'Philadelphia College of Osteopathy',
    },
    {
        'Piracicaba School of Dentistry, Brazil',
        'Piracicaba School of Dentistry',
    },
    {
        'Politechnika Warszawska, Poland',
        'Politechnika Warszawska',
    },
    {
        'Politecnico di Milano',
        'Polytechnic of Milano',
        "Milan Institute of Technology",
        "Milan Polytechnic University",
        'Polytechnic University Of Milan',
    },
    {
        'Polytechnic Institute, Romania',
    },
    {
        'Polytechnic Institute',
    },
    {
        'Polytechnic University Of Turin, Italy',
        'Polytechnic University Of Turin',
    },
    {
        'Polytechnic University, Bucharest',
        'Polytechnic University of Bucharest (Romania)',
    },
    {
        'Ponce School of Medicine, Ponce',
        'Ponce Health Sciences University',
    },
    {
        'Pontifica University Xaveriana',
        'Pontificia Universidad Javeriana',
    },
    {
        'Punjab Agricultural University, India',
        'Punjab Agricultural University',
    },
    {
        'RMIT University',
        'RMIT University, Melbourne, Australia',
    },
    {
        'Rajshahi Medical College',
        'Rajshahi Medical College,  Bangladesh',
    },
    {
        'Rhein-Wesfalische Technische Hochschule, Aachen',
        'RWTH Aachen University',
    },
    {
        'Rhodes University',
        'Rhodes University, South Africa',
    },
    {
        'Rocky Mountain University',
        'Rocky Mountain University of Health Professions, Provo, UT',
    },
    {
        'Rosalind Franklin University',
        'Chicago Medical School, Chicago, IL',
    },
    {
        'Royal College of Surgeons',
        'Royal College of Surgeons, Ireland',
    },
    {
        'Rush University',
        'Rush Medical College',
    },
    {
        'Russian State Medical University, Moscow, Russia',
        'Russian State Medical University',
    },
    {
        'Saarland University',
        'Saarland University (Germany)',
    },
    {
        'Sao Paulo State University',
        'Sao Paulo State University, Brazil',
    },
    {
        'Saratov State University',
        'Saratov State University, Russia',
    },
    {
        'School of Medicine Carol Davila',
        'School of Medicine Carol Davila, Romania',
    },
    {
        'School of Medicine Chiba University, Japan',
        'School of Medicine Chiba University',
    },
    {
        'School of Oriental and African Studies',
        'School of Oriental and African Studies (SOAS), University of London',
    },
    {
        'Shanghai Institute of Optics and Fine Mechanics, CAD',
        'Changchun Institute Of Optics, Fine Mechanics And Physics',
    },
    {
        'Shanghai Institute of Plant Physiology',
        'Shanghai Institute of Plant Physiology, Chinese Academy of Science',
    },
    {
        'Shemyakin Ovchinnikov Institute of Bioorganic Chemistry, Moscow Russia',
        'Chemistry Shemyakin Institute',
    },
    {
        'Shimane Medical University, Japan',
        "Shimane Medical University",
        'Shimane University',
    },
    {
        'Shizuoka University, Japan',
        'Shizuoka University',
    },
    {
        'Sir Wilfrid Laurier University, Canada',
        'Sir Wilfrid Laurier University',
    },
    {
        'Southern National University, Argentina',
        'Southern National University',
    },
    {
        'Southwestern Oklahoma State University College of Pharmacy, Weatherford',
        'Southwestern Oklahoma State University',
    },
    {
        'Space Research Institute (Iki) Ussr Academy Of Sciences',
        'Space Research Institute',
    },
    {
        'Spalding University',
        'Spalding University, Kentucky',
    },
    {
        'Sri Venkateswara University, India',
        'Sri Venkateswara University',
    },
    {
        'St Cyril and Methodius University',
        'Ss. Cyril and Methodius University, Skopje',
    },
    {
        'State Optical Institute',
        'State Optical Institute, Russia',
    },
    {
        'State University of Campinas, Brazil',
        'State University of Campinas',
    },
    {
        'State University of Utrecht',
        'Utrecht University',
        "Rijksuniversiteit te Utrecht, Nederland",
    },
    {
        'Strathclyde University, Scotland',
        'Strathclyde University',
    },
    {
        'TU Bergakademie Freiberg, Germany',
        'TU Bergakademie Freiberg',
    },
    {
        'Technical University Berlin',
        'TU Berlin',
    },
    {
        'Technical University Kaiserslautern, Germany',
        'Technical University Kaiserslautern',
    },
    {
        'Technical University Stuttgart, Germany',
        'Technical University Stuttgart',
    },
    {
        'Technical University of Lisbon, Portugal',
        'Technical University of Lisbon',
    },
    {
        'Technical University of Madrid',
        "Polytechnic University of Madrid",
    },
    {
        'ETSAM Madrid University School of Architecture',
    },
    {
        'Technical University of Poznan',
        'Technical University of Poznan, Poland',
        "Poznan University of Technology",
    },
    {
        'Technion Israel Institute of Technology',
        'Israel Institute of Technology',
    },
    {
        'The College at Brockport, State University of New York',
        'State University of New York, Brockport',
    },
    {
        'The John Marshall Law School, Chicago',
        'John Marshall Law School',
    },
    {
        'The School of the Art Institute of Chicago',
        'Art Institute of Chicago',
    },
    {
        'The Weizmann Institute',
        'The Weizmann Institute of Science',
    },
    {
        'Thomas Jefferson University',
        'Jefferson Medical College',
        "efferson Medical College",
    },
    {
        'United States International University',
        'United States International University - Africa',
    },
    {
        'Universidad Catolica',
        'Universidad Catolica, Chile',
    },
    {
        'Universidad Central del Caribe School of Medicine',
        'Universidad Central del Caribe',
    },
    {
        'Universidad Central del Ecuador, Ecuador',
        'Universidad Central del Ecuador',
    },
    {
        'Universidad Complutense de Madrid',
        'Central University of Madrid',
    },
    {
        'Universidad De Concepci_n',
        'Universidad De ConcepciÜn, Chile',
    },
    {
        'Universidad Del Noreste',
        'Universidad Del Noreste, Mexico',
    },
    {
        'Universidad Francisco Marroquin, Guatemala',
        'Universidad Francisco Marroquin',
    },
    {
        'Universidad Nacional Autonoma',
        'Universidad Nacional Autonoma, Nicaragua',
    },
    {
        'Universidad Nacional De Cuyo',
        'Universidad Nacional De Cuyo, Argentina',
    },
    {
        'Universidad Pontifica Bolivariana, Colombia',
        'Universidad Pontifica Bolivariana',
    },
    {
        'Universidad de El Salvador',
        'University of El Salvador',
    },
    {
        'Universidad del Bio Bio',
        'Universidad del Bio Bio, Chile',
    },
    {
        'Universidade Gama Filho, Brazil',
        'Universidade Gama Filho',
    },
    {
        'Universidade do Rio de Janeiro',
        'University of Rio de Janerio',
    },
    {
        'Universitat Jaume I',
        'Universitat Jaume I, Spain',
    },
    {
        'Universidad de Valencia, Spain',
        'University of Valencia',
        'Universitat de Valencia, Spain',
        'Universitat de Valencia',
    },
    {
        'Universitat zu Koln',
        'University of Koln, Germany,',
    },
    {
        'Universite de Franche Comte, Besancon, France',
        'University of Franche',
    },
    {
        'University Aix-Marseilles II',
        "University of Paris II, Pantheon-Assas",
        'University Aix-Marseilles II, France',
        "University Paris II Pantheon-Assas",
    },
    {
        'University Joseph Fourier',
        'University Joseph Fourier, Grenoble, France',
    },
    {
        'University Of Complutense, Spain',
        'University Of Complutense',
    },
    {
        'University Of Constance, Germany',
        'University Of Constance',
    },
    {
        'University Of Damascus',
        'Damascus University, Syria',
        "Damascus University",
    },
    {
        'University Of North Wales',
        'University Of North Wales, United Kingdom',
    },
    {
        'University Of Osteopathic Medicine And Health Science, Des Moines',
        'Des Moines University',
    },
    {
        'University of Aix-Marseille III, France',
        'University of Aix-Marseille III',
    },
    {
        'University of Aligarh, India',
        'University of Aligarh',
        "A. M. University, Aligarh, India",
    },
    {
        'University of Aston',
        'Aston University',
        'University of Aston',
        'University of Aston, Birmingham',
        "the University of Aston",
    },
    {
        'University of Athens',
        'National and kapodistrian University of Athens',
    },
    {
        'University of Augsburg',
        'University of Augsburg, Germany',
    },
    {
        'University of Bahia',
        'University of Bahia, Brazil',
        "Federal University of Bahia",
    },
    {
        'University of Bari',
        'University of Bari, ITALY',
    },
    {
        'University of Bielefeld',
        'University of Bielefeld, Germany',
    },
    {
        'University of Brasilia',
        'University of Brasilia, Brazil',
    },
    {
        'University of Brescia',
        'University of Brescia, Italy',
        "Brescia University",
    },
    {
        'University of Buckingham, England',
        'University of Buckingham',
    },
    {
        'University of Budapest, Hungary',
        'University of Budapest',
        "University Budapest",
    },
    {
        'University of Cadiz, Spain',
        'University of Cadiz',
    },
    {
        'University of Cagliari, Italy',
        'University of Cagliari',
    },
    {
        'University of Campianas, Sao Paolo',
        'University of Campinas',
    },
    {
        'University of Catania',
        'University of Catania, Italy',
    },
    {
        'University of Cincinnati, The',
    },
    {
        'Cincinnati College-Conservatory of Music',
    },
    {
        'University of Costa Rica',
        'University of Costa Rica, Faculty of Medicine',
    },
    {
        'University of Darmstadt, Germany',
        'University of Darmstadt',
    },
    {
        'University of Derby',
        'University of Derby, Derbyshire College of Higher Education, Derby, UK',
    },
    {
        'University of Dhaka, Bangladesh',
        'University of Dhaka',
    },
    {
        'University of Duisburg-Essen, Germany',
        'University of Duisburg-Essen',
    },
    {
        'University of Essen',
        "Essen University",
        'University of Essen, Germany',
    },
    {
        'University of Extremadura',
        'University of Extremadura, Spain',
    },
    {
        'University of Geneva',
        'universite de Geneve',
    },
    {
        'University of Granada, Spain',
        'University of Granada',
    },
    {
        'University of Graz',
        'University of Graz, Austria',
    },
    {
        'University of Hamburg',
        'Hamburg University, Germany',
    },
    {
        'University of Hawaii',
        'UNIVERSITY OF HAWAII SYSTEM',
    },
    {
        'University of Hohenheim',
        'University of Hohenheim, Germany',
    },
    {
        'University of Huddersfield',
        'University of Huddersfield, Yorkshire, United Kingdom',
    },
    {
        'University of Ibadan, Nigeria',
        'University of Ibadan',
    },
    {
        'University of Ife, Nigeria',
        'University of Ife',
    },
    {
        'University of Ioannina',
        'University of Ioannina, Greece',
    },
    {
        'University of Jabalpur',
        'University of Jabalpur, India',
    },
    {
        'University of Jaen',
        'University of Jaen, Spain',
    },
    {
        'University of Johannesburg, South Africa',
        'University of Johannesburg',
    },
    {
        'University of Jyvaskyla, Finland',
        'University of Jyvaskyla',
    },
    {
        'University of Kaiserslautern, Germany',
        'University of Kaiserslautern',
    },
    {
        'University of Kanpur, India',
        'University of Kanpur',
    },
    {
        'University of Khartoum',
        'University of Khartoum, Sudan',
    },
    {
        'University of La Plata',
        'University of La Plata, Argentina',
    },
    {
        'University of Lagos, Nigeria',
        'University of Lagos',
    },
    {
        'University of Leoben, Austria',
        'University of Leoben',
    },
    {
        'University of Leon, Spain',
        'University of Leon',
    },
    {
        'University of Lethbridge, Alberta, Canada',
        'University of Lethbridg',
    },
    {
        'University of Limerick, Ireland',
        'University of Limerick',
    },
    {
        'University of Mainz',
        "Mainz University",
        'Johannes Gutenberg University',
    },
    {
        'University of Malaya, Kuala Lumpur, Malaysia',
        'University of Malaya',
    },
    {
        'University of Mannheim',
        'University of Mannheim, Germany',
    },
    {
        'University of Marseille',
        'University of Marseille, France',
    },
    {
        'University of Mexico',
        'National Autonomous University of Mexico',
    },
    {
        'University of Milano',
        'University of Milano, Italy',
        "Universita degli Studi, Milan, Italy",
        "State University of Milan",
        "Universita’ degli Studi Di Milano",
        'University Di Milano',
        'University of Milan',
    },
    {
        'University of Modena, Italy',
        'University of Modena',
    },
    {
        'University of Namur, Belgium',
        'University of Namur',
    },
    {
        'University of Nancy',
        'University of Nancy, France',
    },
    {
        'University of Napoli Federico II, Italy',
        'University of Napoli Federico II',
    },
    {
        'University of Neuchatel',
        'University of Neuchatel, Switzerland',
    },
    {
        'University of New Brunswick',
        'University of New Brunswick, Fredericton',
    },
    {
        'University of Nigeria, Nsukka',
        'University of Nigeria',
    },
    {
        'University of Oldenburg, Germany',
        'University of Oldenburg',
    },
    {
        'University of Orleans',
        'University of Orleans, France',
    },
    {
        'University of Panama, School of Medicine Panama',
        'University of Panama',
    },
    {
        'University of Paris I (Pantheon-Sorbonne)',
        'University of Paris 1 Pantheon-Sorbonne (France)',
        'EHESS, Sorbonne Pantheon University Paris',
    },
    {
        'University of Paris XIII (Nord/North)',
        'University of Paris-Nord (France)',
    },
    {
        'University of Paris-X',
        'University of Paris X (Nanterre)',
    },
    {
        'University of Pasteur Strasbourg, France',
        'University of Pasteur Strasbourg',
    },
    {
        'University of Poona, India',
        'University of Poona',
    },
    {
        'University of Port Elizabeth, South Africa',
        'University of Port Elizabeth',
    },
    {
        'University of Porto, Portugal',
        'University of Porto',
    },
    {
        'University of Queensland',
        'University of Hong Kong',
    },
    {
        'University of Rosario, Argentina',
        'University of Rosario',
    },
    {
        'University of Rouen',
        'University of Rouen, France',
    },
    {
        'University of Salamanca, Spain',
        'University of Salamanca',
    },
    {
        'University of Salonica, Greece',
        'University of Salonica',
    },
    {
        'University of Santo Tomas, Philippines',
        'University Of Santo Tomas',
    },
    {
        'University of Sao Paulo',
        'University of San Paulo',
    },
    {
        'University of Sassari, Italy',
        'University of Sassari',
    },
    {
        'University of Science and Technology of China',
    },
    {
        'Sun Yat-sen University',
    },
    {
        'University of Seville',
        'University of Seville, Spain',
    },
    {
        'University of South Bohemia, Czech Republic',
        'University of South Bohemia',
    },
    {
        'University of Strathclyde, Scotland',
        'University of Strathclyde, Glasgow',
    },
    {
        'University of Technology of Troyes, France',
        'University of Technology of Troyes',
    },
    {
        'University of Tehran Medical School',
        'Tehran University of Medical Sciences',
    },
    {
        'University of The Free State, South Africa',
        'University of The Free State',
    },
    {
        'University of Timisoara',
        'University of Timisoara, Romania',
    },
    {
        'University of Tokushima, Japan',
        'University of Tokushima',
    },
    {
        'University of Trento, Italy',
        'University of Trento',
    },
    {
        'University of Tucuman',
        'University of Tucuman, Argentina',
    },
    {
        'University of Ulm, Germany',
        'University of Ulm',
    },
    {
        'University of Valladolid, Spain',
        'University of Valladolid',
        "Universidad de Valladolid",
        "Universidad de Valladolid, Spain",
    },
    {
        'University of Veterinary Science, Budapest',
        'University of Veterinary Science, Budapest, Hungary',
    },
    {
        'University of Wales, Bangor',
        'University of Wales',
    },
    {
        'University of West Florida',
        'University of West Florida, Pensacola, FL',
    },
    {
        'University of West Indies',
        'University of West Indies, Jamaica',
    },
    {
        'University of Wollongong',
        'University of Wollongong, Australia',
    },
    {
        'University of Zaragoza, Spain',
        'University of Zaragoza',
    },
    {
        'University of the Basque Country',
        'University of Basque Country, Vizcaya',
    },
    {
        'University, Budapest, Hungary',
        'Corvinus University of Budapest',
    },
    {
        'Université Joseph Fourier',
        'Universite Joeseph Fourier, Grenoble, France',
    },
    {
        'Vakhtangov Academy Theater, Russia',
        'Vakhtangov Academy Theater',
    },
    {
        'Vilnius University, Lithuania',
        'Vilnius University',
    },
    {
        'Visva Bharati University',
        'Visva Bharati University, India',
    },
    {
        'Voronezh State University',
        'Voronezh State University, Russia',
        "Voronezh University",
        "University of Voronezh",
        "University of Voronezh, Russia",
    },
    {
        'Warnborough College, Ireland',
        'Warnborough College',
    },
    {
        'Washburn University',
        'Washburn University, Topeka',
    },
    {
        'Western University',
        'Western University, Canada',
    },
    {
        'Widener University - Chester',
        'Widener University',
    },
    {
        'Worcester Polytechnic Institute',
    },
    {
        'Institute Biomedical Engineering',
    },
    {
        'Wright Institute',
        'Wright Institute, Berkeley, CA',
    },
    {
        'Wuerzburg University',
        'Wuerzburg University, Germany',
    },
    {
        'Zagazig University',
        'Zagazig University, Egypt',
    },
]

ALL_ALIASES = EMPLOYMENT_MERGING_ALIASES + LONG_ALIASES + TOKEN_MATCHED_ALIASES + TWO_LENGTH_ALIASES

################################################################################
# LOW CONFIDENCE aliases
################################################################################

# these aliases are ones where there is an exact match to one of the aliases
# above when set sorting
SET_MATCHED_DEGREE_ALIASES = [
    {
        'Helsinki University',
        'University of Helsinki',
    },
    {
        'Iasi University',
        'University of Iasi',
    },
    {
        'Vienna University of Technology',
        'University of Vienna',
    },
    {
        'Canterbury University',
        'University of Canterbury',
    },
    {
        'Fourth Military Medical University',
        'Fourth Military Medical University, China',
    },
    {
        'Paul Sabatier University',
        'University Paul Sabatier',
    },
    {
        'University of Goethe, Frankfurt',
        'Goethe University, Frankfurt',
    },
    {
        'Academy of Sciences of the USSR',
        'USSR Academy of Sciences',
        "Academy of Sciences, St. Petersburg"
    },
    {
        'Leningrad University',
        'University of Leningrad',
    },
    {
        'Wuerzburg University, Germany',
        'University of Wuerzburg,',
    },
    {
        'Cambridge University',
    },
    {
        'Cambridge College',
    },
    {
        'University of Wales',
        'University College Of Wales',
    },
    {
        'University of Kanpur, India',
        'Kanpur University',
    },
    {
        'University of the Pacific',
        'Pacific University',
    },
    {
        'University of Goteborg, Goteborg, Sweden',
        'University of Goteborg, Sweden',
    },
    {
        'Lady Hardinge Medical College',
        'Lady Hardinge Medical College, India',
    },
    {
        'University Of Complutense',
        'Complutense University',
    },
    {
        'University of Twente',
        'Twente University',
    },
    {
        'University of the Arts London',
        'University of the Arts, London',
    },
    {
        'Indian Institute of Chemical Technology',
        'Indian Institute Of Chemical Technology',
    },
    {
        'Medical School Of Hannover',
        'Hannover Medical School, Hannover',
    },
    {
        'Istanbul University',
        'University of Istanbul',
    },
    {
        'University of Bradford',
        'Bradford University',
    },
    {
        'State University of New York Empire State College',
        'Empire State College, New York',
    },
    {
        'University of Washington',
        'Washington College',
    },
    {
        'Technical University Berlin',
        'Berlin Technical University',
    },
    {
        'University of Science and Technology of China',
        'University of Science and Technology',
    },
    {
        "St. John's University",
        "St. John's College",
    },
    {
        'National University of Ireland',
        'the National University of Ireland',
    },
    {
        'Chongqing University',
        'Chongqing University, China',
    },
    {
        'University of Bath',
        'Bath University',
    },
    {
        'Attila Jozsef University Szeged',
        "Attila Jozsef University Szeged, Hungary",
        'Jozsef Attila University',
        "ozsef Attila Universit",
        'Szeged University',
        'University of Szeged',
        'Szeged University, Hungary',
        'Szeged University',
        "JATE Szeged University",
        "JATE University of Szeged,Hungary",
    },
    {
        "Medical University Szeged",
    },
    {
        "Medical University of Bialystok",
        "Medical University of Bialystok, Poland",
        "Academy of Medicine, Bialystok",
    },
    {
        'Aleppo',
        'University of Aleppo',
    },
    {
        'State University of New York',
        'University of the State of New York',
    },
    {
        'The School of the Art Institute of Chicago',
        'School of the Art Institute of Chicago',
    },
    {
        'MD Anderson Cancer Center (Texas)',
        'University of Texas MD Anderson Cancer Center, The',
    },
    {
        'New School for Social Research',
        'New School for Social Research, The',
    },
]

# this set of aliases was identified by looking at institution names that,
# when removing things like country, and 'of' and etc, matched
DUPLICATES_ALIASES = [
    {
        'University of Rome',
        'Rome University',
    },
    {
        'University of Aix-Marseille III',
        'Aix-Marseille University',
        'University of Aix-Marseille',
    },
    {
        'University of Philippines',
        'University of the Philippines',
    },
    {
        'Moscow University',
        'University of Moscow,',
    },
    {
        "University of St. Petersburg",
        "University of St. Petersburg, Russia"
        'St. Petersburg State University',
        'St Petersburg State University',
        "St.Petersburg State University",
    },
    {
        'Wroclaw University of Technology',
    },
    {
        'University of Wroclaw',
    },
    {
        'The Open University',
        'Open University',
    },
    {
        'Union College',
    },
    {
        'Union University',
    },
    {
        'Chinese Academy of Sciences',
        'University of Chinese Academy of Sciences',
    },
    {
        'WILLIAMS COLLEGE',
        'Williams College',
    },
    {
        'Henan Medical University, Henan, China',
        'Henan Medical University',
    },
    {
        'Capital University of Medical Sciences',
        "Beijing Capital Medical University",
        'Capital University of Medical Sciences, China',
    },
    {
        'National University At Rosario',
        'National University, Rosario',
        'National University of Rosario',
    },
    {
        'Technical University Chemnitz, Germany',
        'Technical University Chemnitz',
    },
    {
        'Pratt Institute',
        'Pratt Institute of Technology',
    },
    {
        'Shanxi Medical University',
        'Shanxi Medical University, China',
    },
    {
        'Kilpauk Medical College',
        'Kilpauk Medical College, India',
    },
    {
        'School of Medicine of the University of Sao Paulo, Sao Paulo',
        'Sao Paulo University School of Medicine',
    },
    {
        'Harbin Medical University, China',
        'Harbin Medical University',
    },
    {
        'Berlin University of the Arts, Germany',
        'Berlin University of the Arts',
    },
    {
        'Goa Medical College, India',
        'Goa Medical College',
    },
    {
        'Changchun Institute of Applied Chemistry',
        'Changchun Institute Of Applied Chemistry, China',
    },
    {
        'Institute of Molecular Biology, Moscow',
        'W. Engelhardt Institute of Molecular Biology in Moscow, Russia',
        'W. Engelhardt Institute of Molecular Biology in Moscow',
    },
    {
        'Universitat zu Lubeck, Germany',
        'Universitat zu Lubeck',
    },
    {
        "Tokyo Women's Medical College",
        "Tokyo Women's Medical University",
    },
    {
        'Karlsruhe University, Germany',
        'Karlsruhe University',
        'Karlsruhe University, Germany',
        'Karlsruhe Institute of Technology, Germany',
        'Karlsruhe Institute of Technology',
        'Karlsruhe University, Germany',
        'University of Karlsruhe',
        'Technical University of Karlsruhe, Germany',
        'Universitat Friedericiana Karlsruhe',
    },
    {
        'University of Zagreb',
        'Zagreb University',
    },
    {
        'Calcutta University',
        'University of Calcutta',
    },
    {
        'Haifa University',
        'University of Haifa',
    },
    {
        'Adelaide University',
        'University of Adelaide',
    },
    {
        'Curtin University',
        'Curtin University of Technology',
    },
    {
        'Annamalai University, India',
        'Annamalai University',
    },
    {
        'MACALESTER COLLEGE.',
        'Macalester College',
    },
    {
        'University of Magdeburg',
        'University of Magdeburg, Germany',
    },
    {
        'Sardar Patel University, India',
        'Sardar Patel University',
    },
    {
        'VASSAR COLLEGE',
        'Vassar College',
    },
    {
        'Regis University',
        'Regis College',
    },
    {
        'Witten/Herdecke University, Germany',
        'Witten/Herdecke University',
    }
]

# This is everything else. ugly here.
MISCELLANEOUS_ALIASES = [
    {
        '1st Moscow medical institute',
        'First Moscow State Medical University',
    },
    {
        'A. T. Still University',
    },
    {
        'Academia Sinica, Beijing',
    },
    {
        'Academy of Art University in San Francisco',
    },
    {
        'Academy of Military Medical Sciences',
    },
    {
        'Academy of Sciences of Moldova',
    },
    {
        'Academy of Sciences of Ukraine',
    },
    {
        'Acharya Nagarjuna University',
        "Acharya Nagarjuna University, India"
    },
    {
        'Addis Ababa University',
    },
    {
        'Aga Khan University, Karachi, Pakistan',
    },
    {
        'Ajou University, South Korea',
        "Ajou University, Korea",
    },
    {
        'Al Azhar University',
    },
    {
        'Albany College of Pharmacy and Health Sciences',
    },
    {
        'Albany Law School, New York',
    },
    {
        'Albert Ludwigs University',
    },
    {
        'Albert Szent-Gyorgyi Medical University',
    },
    {
        'Alcala De Henares University, Madrid, Spain',
    },
    {
        'Alderson Broaddus College',
    },
    {
        'Aligarh Muslim University',
    },
    {
        'Allama Iqbal Medical College',
    },
    {
        'American Conservatory Theater, San Francisco',
    },
    {
        'American Film Institute',
    },
    {
        'American University Of Caribbean, British West Indies',
    },
    {
        'American University of Antigua',
    },
    {
        'American University of the Caribbean, Miami',
    },
    {
        'American University, Cairo',
    },
    {
        'Anglia Polytechnic University',
    },
    {
        'Anglia Ruskin University',
    },
    {
        'Anna University',
        'Anna University, Chennai',
    },
    {
        'Antioch School of Law',
    },
    {
        'Arcadia University',
    },
    {
        'Argosy University, Atlanta',
    },
    {
        'Argosy University, Sarasota',
    },
    {
        'Armenian National Academy of Sciences (NAS)',
    },
    {
        'Art Center College of Design In Pasadena, Calif',
    },
    {
        'Ashland University',
    },
    {
        'Assiut University',
    },
    {
        'Atlantic Veterinary College - University of Prince Edward Island',
    },
    {
        'Autonomous University of Guadalajara',
    },
    {
        'Autonomous University of Madrid',
    },
    {
        'Autonomous University of Nuevo, Leon',
    },
    {
        'Autonomous University',
    },
    {
        'B. J. Medical College',
    },
    {
        'Babasaheb Bhimrao Ambedkar University',
    },
    # not the same as above
    {
        "Dr. B.R. Ambedkar University",
        "Dr. B.R. Ambedkar University, India",
        "Dr.B.R.Ambedkar University",
    },
    {
        'Bach Institute of Biochemistry Moscow',
    },
    {
        'Bangalore Medical College',
    },
    {
        'Bangor University',
    },
    {
        'Bank Street College of Education',
    },
    {
        'Baroda Medical College',
        "Medical College Baroda",
    },
    {
        'Bauhaus University, Weimar',
    },
    {
        'Bauman Moscow State Technical University',
    },
    {
        'Beihang University',
    },
    {
        'Beijing Medical University',
    },
    {
        'Beijing University',
    },
    {
        'Belarus Academy of Sciences',
    },
    {
        'Bellarmine University',
    },
    {
        'Ben-Gurion University of the Negev',
    },
    {
        'Bennington Writing Seminars',
    },
    {
        'Berklee College of Music',
    },
    {
        'Biotechnology, Royal Institute of Technology  (KTH), Sweden',
    },
    {
        'Birkbeck College',
    },
    {
        'Blaise-Pascal University',
    },
    {
        'Bogomoletz Institute of Physiology',
    },
    {
        'Bose Institute, Calcutta',
    },
    {
        'Boston Conservatory',
    },
    {
        'Boston State University',
    },
    {
        "Bournemouth University",
        'Bournemouth University, UK',
    },
    {
        'Bradley University',
    },
    {
        'Bridgewater College',
    },
    {
        'Brunel University',
    },
    {
        'Budapest University of Technology & Economics',
    },
    {
        'Butler University',
    },
    {
        'Byelorussian Academy of Sciences',
    },
    {
        'Calcutta National Medical College',
        "Calcutta National Medical College, India"
    },
    {
        'Calicut Medical College',
    },
    {
        'California College of Arts and Crafts',
    },
    {
        'California College of the Arts',
    },
    {
        'California School of Professional Psychology, Fresno',
    },
    {
        'California School of Professional Psychology, Los Angeles',
    },
    {
        'California School of Professional Psychology, San Diego',
    },
    {
        'California School of Professional Psychology, San Francisco',
    },
    {
        'California State University, Dominguez Hills',
    },
    {
        'California State University, East Bay',
    },
    {
        'California State University, Los Angeles',
    },
    {
        'California State University, San Bernardino',
    },
    {
        'California University of Pennsylvania',
    },
    {
        'California Western School of Law',
    },
    {
        'Campbell University',
    },
    {
        'Cancer Research Center, Moscow',
    },
    {
        'Canisius College',
    },
    {
        'Capella University',
    },
    {
        'Capital Institute of Medicine',
    },
    {
        'Carleton University, Ottawa, Canada',
    },
    {
        'Carlos Albizu University',
    },
    {
        'Carol Davila University of Medicine',
    },
    {
        'Catholic University School of Medicine, Rome',
    },
    {
        'Catholic University of Korea',
    },
    {
        'Catholic University of Louvain',
    },
    {
        'Catholic University of Milan',
    },
    {
        'Catholic University of Nijmegen',
    },
    {
        'Catholic University of Rome',
    },
    {
        'Catholic University of the Sacred Heart, Rome',
    },
    {
        'Center for Advanced Studies',
    },
    {
        'Central Connecticut State University',
    },
    {
        'Central Drug Research Institute',
    },
    {
        'Central Queensland University',
    },
    {
        'Central Saint Martins College of Art and Design, London',
    },
    {
        'Central School of Speech and Drama, and Drama Centre London',
    },
    {
        'Central South University',
    },
    {
        'Central St. Martins College of Art and Design, London',
    },
    {
        'Centro de Investigaciones',
    },
    {
        'Chalmers University of Technology',
    },
    {
        'Chapman University',
    },
    {
        'Charite University Medicine',
    },
    {
        'Charles III University of Madrid',
    },
    {
        'Charles University in Prague',
    },
    {
        'Charles de Gaulle University, Lille III, France',
    },
    {
        'Chatham University',
    },
    {
        'Chaudhary Charan Singh University, Meerut',
    },
    {
        'Chelsea Art College',
    },
    {
        'Chiba University, Japan',
    },
    {
        'Chicago School of Professional Psychology, Chicago Campus',
    },
    {
        'Chicago-Kent College of Law',
    },
    {
        'China Medical University',
    },
    {
        'China Pharmaceutical University, Nanjing, China',
    },
    {
        'Chinese Academy of Preventative Medicine',
    },
    {
        'Chinese Center for Disease Control and Prevention, China',
    },
    {
        'Chongqing Medical University',
    },
    {
        'Christ Seminary Seminex, St. Louis',
    },
    {
        'Christian Albrechts University',
        'Christian-Albrechts University at Kiel',
    },
    {
        'Chulalongkorn University',
    },
    {
        'Chungbuk National University',
    },
    {
        'Clarion University of Pennsylvania',
    },
    {
        'Clausthal University of Technology in Germany',
    },
    {
        'Cleveland Clinic',
    },
    {
        'Cleveland-Marshall College of Law',
    },
    {
        'Cochin University of Science and Technology in india',
    },
    {
        'Coimbatore Medical College',
    },
    {
        'College de France',
    },
    {
        'Colorado College',
    },
    {
        'Colorado Technical University',
    },
    {
        'Columbia College of Physicians and Surgeons',
    },
    {
        'Comenius University',
    },
    {
        'Computing Center of the Academy of Sciences, Moscow, Russia',
    },
    {
        'Concordia University',
    },
    {
        'Cooper Union',
    },
    {
        'Corpus Christi State University',
    },
    {
        'Council of National Academic Awards. London, UK',
    },
    {
        'Courant Institute of Mathematical Sciences',
        'Courant Institute',
    },
    {
        'Covenant Theological Seminary',
    },
    {
        'Coventry University',
    },
    {
        'Cranbrook Schools, Bayside',
    },
    {
        'Cranfield Institute of Technology',
    },
    {
        'Cranfield University',
    },
    {
        'Cukurova University',
    },
    {
        'Curtis Institute of Music',
    },
    {
        'Czechoslovak Academy of Sciences',
    },
    {
        'Dalian Medical University',
    },
    {
        'Dalian University of Technology (China)',
        "Dalian University of Technology",
    },
    {
        'De Montfort University',
    },
    {
        "Dell'Arte International School of Physical Theatre",
    },
    {
        'Delta State University',
    },
    {
        'Dickinson College',
    },
    {
        'Dongguk University',
    },
    {
        'Dowling College',
    },
    {
        'Dublin Institute of Technology',
    },
    {
        'ENPC, Paris',
    },
    {
        'East China University of Science and Technology, Shanghai',
    },
    {
        'East Stroudsburg University',
    },
    {
        'East Texas State University',
    },
    {
        'Eastern New Mexico University',
    },
    {
        'Eastern University',
    },
    {
        'Eastern Virginia Medical School',
    },
    {
        'Eastern Washington University',
    },
    {
        'Ecole Centrale Lyon',
        "Ecole Centrale de Lyon",
    },
    {
        'Ecole Nationale Superieure des Telecommunications',
    },
    {
        'Ecole Nationale Supérieure en Génie des Technologies Industrielles',
    },
    {
        'Ecole Normale Superieure de Lyon',
        'Ecole Normale Superieure',
        'Ecole Normale Superieure, France',
    },
    {
        'Ecole Polytechnique Federale de Lausanne',
        'Ecole Polytechnique',
    },
    {
        'Ecole des Hautes Etudes (EPHE)',
    },
    {
        'Ecole des Hautes Etudes en Sciences Sociales (EHESS)',
        'Ecoles des Hautes Etudes en Sciences Sociales',
        "Ecole Pratique des Hautes Etudes en Sciences Sociales"
    },
    {
        'Edinboro University of Pennsylvania, Edinboro',
    },
    {
        'Edith Cowan University',
    },
    {
        'Ehime University',
    },
    {
        'El Colegio de Mexico',
    },
    {
        'Elmira College',
    },
    {
        'Embl Heidelberg Germany',
    },
    {
        'Engelhardt Institute of Molecular Biology',
    },
    {
        'Espirito Santo Federal University Medical School, Brazil.',
    },
    {
        'Europa Universität Viadrina',
    },
    {
        'European Molecular Biology Laboratory',
    },
    {
        'Evergreen State College, Olympia, WA',
    },
    {
        'Eötvös Loránd University',
    },
    {
        'Far Eastern University, Philippines',
    },
    {
        'Fashion Institute of Technology',
    },
    {
        'Favaloro University',
    },
    {
        'Federal Institute of Technology, Lausanne, Switzerland',
    },
    {
        'Federal University of Sao Paulo',
    },
    {
        'Federico II University,Naples, Italy',
        'University of Naples Federico II',
        'University of Naples',
        "University Di Napoli",
        "University of Napoli, Italy",
    },
    {
        "University of Naples Parthenope, Italy",
    },
    {
        'Francois Rabelais University',
        "University Francois Rabelais",
    },
    {
        'Franklin Pierce Law Center',
    },
    {
        'Franz Liszt Academy of Music, Budapest, Hungary',
    },
    {
        'Frontier Nursing University',
    },
    {
        'Fudan University',
    },
    {
        'Fujian Medical University',
    },
    {
        'Fukui University',
    },
    {
        'Full Sail University',
    },
    {
        'Fuller Graduate School of Psychology, Pasadena, CA',
    },
    {
        'Gandhi Medical College',
        'Gandhi Medical College, Hyderabad, India',
    },
    {
        'Ganesh Shankar Vidyarthi Memorial Medical College',
    },
    {
        'Gardner-Webb University',
    },
    {
        'Gauhati University, Gauhati',
    },
    {
        'Gazi Üniversitesi',
    },
    {
        'Georgian Academy of Sciences',
        'Georgian Academy of Sciences, Georgia',
    },
    {
        'Gerhard-Mercator University, Duisburg, Germany',
    },
    {
        'German Cancer Research Center',
    },
    {
        'Gifu University',
    },
    {
        'Goa University',
    },
    {
        'Goethe Universitä',
    },
    {
        'Golden Gate University',
    },
    {
        'Goldsmiths College',
    },
    {
        'Gordon-Conwell Theological Seminary',
    },
    {
        'Government Medical College Chandigarh Punjab University',
    },
    {
        'Government Medical College Srinagar, Jammu and Kashmir, India',
    },
    {
        'Government Medical College, Guru Nanak Dev University, Amritsar, India',
    },
    {
        'Governor?s State University',
    },
    {
        'Graduate University for Advanced Studies',
    },
    {
        'Grand Canyon University',
    },
    {
        'Grant Medical College and Sir JJ Group of Hospitals',
    },
    {
        'Grenoble Ecole de Management',
    },
    {
        'Grigore T. Popa University of Medicine and Pharmacy, Iasi',
    },
    {
        'Guangzhou University of Chinese Medicine, P.R. China',
    },
    {
        'Guildhall School Of Music, London',
        "London Guildhall University",
    },
    {
        'Gujarat University',
    },
    {
        'Gunma University',
    },
    {
        "Guy's and St. Thomas's Hospitals",
    },
    {
        'Gwangju Institute of Science and Technology',
    },
    {
        'Hadassah Medical School',
    },
    {
        'Hamamatsu University',
    },
    {
        'Hanyang University',
    },
    {
        'Harbin Institute of Technology',
    },
    {
        'Haryana Agricultural University',
    },
    {
        'Havana Advanced Institute Of Medical Sciences',
    },
    {
        'Haverford College',
    },
    {
        'Hebei Medical University',
    },
    {
        'Hebrew Union College',
    },
    {
        'Heinrich-Heine University',
    },
    {
        'Helsinki University of Technology (Finland)',
    },
    {
        'Henry Poincare University',
    },
    {
        'Higher Institute of Medical Sciences',
    },
    {
        'Hiroshima University',
    },
    {
        'Hokkaido University',
        "University Of Hokkaido, Japan",
    },
    {
        'Hollins University',
    },
    {
        'Hong Kong Baptist University',
    },
    {
        'Huazhong Agricultural University, China',
    },
    {
        'Huazhong University of Science and Technology',
    },
    {
        'Hubei Medical University Xianning Medical School, China',
    },
    {
        'Humboldt State University',
        'Humboldt University',
    },
    {
        'Hunan Medical University',
    },
    {
        'Hunan University, China',
    },
    {
        'IMTECH',
    },
    {
        'INSA de Rouen at France',
    },
    {
        'Ibero-American University, Mexico',
    },
    {
        'Illinois College of Optometry',
    },
    {
        'Illinois School of Professional Psychology',
    },
    {
        'Illinois Wesleyan University',
    },
    {
        'Imperial Cancer Research Fund',
    },
    {
        'India Institute of Technology, Madras',
    },
    {
        'Indian Agricultural Research Institute',
    },
    {
        'Indian Association for the Cultivation of Science',
    },
    {
        'Indian Institute of Management',
    },
    {
        'Indian Institute of Science',
    },
    {
        'Indian Institute of Technology',
    },
    {
        'Industrial Toxicology Research Center',
    },
    {
        'Insa Toulouse',
    },
    {
        'Insead university',
    },
    {
        'Institut National Polytechnique de Lorraine',
    },
    {
        'Institut National Polytechnique, Toulouse, France',
    },
    {
        "Institut d'Etudes Politiques de Paris",
    },
    {
        'Institut de Physique du Globe de Paris',
    },
    {
        'Institute Of Biochemistry And Biophysics Polish Academy Of Sciences',
    },
    {
        'Institute Of Child Health, University College London',
    },
    {
        'Institute Of Physics, National Academy Of Science, Ukraine',
    },
    {
        'Institute for Clinical Social Work',
    },
    {
        'Institute for Nuclear Research, Moscow',
    },
    {
        'Institute for Theoretical and Experimental Physics, Moscow',
    },
    {
        'Institute of Atomic Physics',
    },
    {
        'Institute of Atomic Physics, Romania',
    },
    {
        'Institute of Biochemistry, Russian Academy of Sciences',
    },
    {
        'Institute of Biology, Russian Academy of Sciences',
    },
    {
        'Institute of Biophysics, Chinese Academy of Sciences',
    },
    {
        'Institute of Child Psychology at the University of Minnesota',
    },
    {
        'Institute of Crystallography, Moscow',
    },
    {
        'Institute of Cytology and Genetics, Novosibirsk, Russia',
        "Novosibirsk Institute of Cytology and Genetics, Russia"
        "Genetics Inst of Cytology & Genetics",
    },
    {
        'Institute of Education',
    },
    {
        'Institute of Fine Arts of New York University',
    },
    {
        'Institute of Genetics and Chinese Academy of Sciences',
    },
    {
        'Institute of Medicine',
    },
    {
        'Institute of Microbial Technology, Chandigarh, India',
    },
    {
        'Institute of Microbiology of the Czech Academy of Sciences',
    },
    {
        'Institute of Molecular Genetics, Moscow',
    },
    {
        'Institute of Neuroscience, Chinese Academy of Sciences',
    },
    {
        'Institute of Physics, Chinese Academy of Sciences, Beijing',
    },
    {
        'Institute of Solid State Physics of Russian Academy of Science',
    },
    {
        'Institute of Theoretical and Experimental Physics, Moscow',
    },
    {
        'Institute of Virology, Chinese Academy of Preventive Medicine, China',
    },
    {
        'Instituto Tecnologico De Santo Domingo',
    },
    {
        'Instituto Tecnologico y de Estudios Superiores de Monterrey',
    },
    {
        'Instituto Venezolano de Investigaciones Cientificas, Caracas, Venezuela',
    },
    {
        'International School For Advanced Studies',
    },
    {
        'Iran University Of Medical Sciences, Iran',
    },
    {
        'Islamic Azad University',
    },
    {
        'Istanbul Technical University',
    },
    {
        'Istituto Universitario Orientale, Napoli',
    },
    {
        'Istituto Universitario di Architettura, Venice',
    },
    {
        'Iwate University Japan',
        "Iwate University",
    },
    {
        "Iwate Medical University",
    },
    {
        'Jacksonville University',
    },
    {
        'Jacobs University Bremen (Germany)',
    },
    {
        'Jadavpur University',
    },
    {
        'Jagiellonian University',
    },
    {
        'James Cook University',
    },
    {
        'Jamia Hamdard university, New Delhi, India',
    },
    {
        'Jamia Millia Islamia University, New Delhi',
    },
    {
        'Japan Advanced Institute of Science and Technology',
    },
    {
        'Jawaharlal Institute of Post-Graduate Medical Education And Research',
        "Jawaharlal Nehru Medical College",
        "Jawaharlal Nehru Medical College, India",
        "awaharlal Nehru University",
        "awaharlal Nehru University, India",
    },
    {
        "Jawaharlal Nehru Technological University, Hyderabad, India",
    },
    {
        'Jawaharlal Nehru University',
    },
    {
        'Jiangnan University',
    },
    {
        "Jiangxi Agricultural University",
    },
    {
        'Jiangxi Medical College',
    },
    {
        'Jiaotong Univeristy',
    },
    {
        'Jichi Medical School, Japan',
        "Jichi Medical School",
    },
    {
        'Jikeii University of Medicine',
    },
    {
        'Jilin University',
    },
    {
        'Jiwaji University',
        "Jiwaji University, India",
    },
    {
        "Jozef Stefan International Postgraduate School",
        "Jozef Stefan International Postgraduate School, Slovenia",
    },
    {
        'Johann Wolfgang Goethe University',
    },
    {
        'Johannes Kepler University',
    },
    {
        'Joint Institute for Nuclear Research',
    },
    {
        'Juilliard School',
    },
    {
        'Juntendo University',
        'Juntendo University, Toyko',
    },
    {
        'Justus Liebig University',
    },
    {
        'Kakatiya Medical College',
    },
    {
        'Kansas City University of Medicine and Biosciences',
    },
    {
        'Kaplan University, Chicago',
    },
    {
        'Karl-Franzens-University',
    },
    {
        'Kasr El-Aini School of Medicine',
    },
    {
        'Katholieke Universiteit, Nijmegan',
    },
    {
        'Kazan State Medical University',
    },
    {
        'Kazan State University',
    },
    {
        'Kean College',
    },
    {
        'Keio University',
    },
    {
        'Keller Graduate School of Management',
    },
    {
        'Kennedy-Western University',
    },
    {
        'Kennesaw State University',
    },
    {
        'Keuka College',
    },
    {
        'Kharkov State University',
    },
    {
        'Kiev Medical Institute, Kiev',
    },
    {
        'Kiev National Taras Shevchenko University',
        'Kiev National Taras Shevchenko University, Ukraine',
        "Shevchenko State University",
        "Kiev Shevchenko University, Kiev, Ukraine",
        "Kyiv National Shevchenko University",
        "Kyiv Taras Shevchenko University",
    },
    {
        'Kiev Polytechnic Institute',
        "Kyiv Polytechnic Institute",
    },
    {
        'Kiev State University',
        'Kiev University',
        "University of Kyiv",
    },
    {
        'Kingston University',
    },
    {
        'Kirksville College of Osteopathic Medicine',
    },
    {
        'Kitasato University',
    },
    {
        'Kobe University',
    },
    {
        'Konkuk University',
    },
    {
        'Korea University',
    },
    {
        'Kurchatov Institute',
    },
    {
        'Kurukshetra University',
    },
    {
        'Kurume University School of Medicine, Kurume',
    },
    {
        'Kyoto Prefectural University of Medicine',
    },
    {
        'Kyoto University',
        "University of Kyoto",
        "University of Kyoto, Japan",
    },
    {
        'Kyung Hee University',
    },
    {
        'Kyunghee University School of Medicine-Korea',
    },
    {
        'Kyungpook National University',
    },
    {
        'Kyushu Institute of Technology',
    },
    {
        'Kyushu University',
    },
    {
        'La Salle University',
    },
    {
        'La Trobe University',
    },
    {
        'Lamar University',
    },
    {
        'Lanzhou Institute of Chemical Physics, Lanzhou ,China',
    },
    {
        'Lanzhou University',
    },
    {
        'Lasalle University, Pennsylvania',
    },
    {
        'Latvian Academy of Science',
    },
    {
        'Latvian Medical Academy',
    },
    {
        'Laval University',
    },
    {
        'Lawrence University',
    },
    {
        'Lebanese University',
    },
    {
        'Lebedev Physical Institute',
    },
    {
        'Leibniz University of Hanover',
    },
    {
        'Leopold-Franzens University',
    },
    {
        'Lewis and Clark College',
    },
    {
        'Liberty University',
    },
    {
        'Lincoln Memorial University',
    },
    {
        'Lithuanian Academy of Sciences, Vilnius',
        "Lithuania Academy of Sciences",
    },
    {
        'London South Bank University',
    },
    {
        'Long Island University',
    },
    {
        'Louis Pasteur University',
    },
    {
        'Louisiana State University Health Sciences Center, New Orleans',
    },
    {
        'Loyola Law School, Los Angeles',
    },
    {
        'Loyola Marymount University',
    },
    {
        'Lubbock Christian University',
    },
    {
        'Ludwig Institute for Cancer Research - University College Branch, London',
    },
    {
        'Lund Institute of Technology',
    },
    {
        'Lynchburg College',
    },
    {
        'Lynn University',
    },
    {
        'M.S. Ramaiah Medical College',
    },
    {
        'MGH Institute of Health Professions',
    },
    {
        'Madurai Kamaraj University',
        "Madurai-Kamaraj (India)",
    },
    {
        'Magdalen College',
    },
    {
        'Maharashtra University of Health Sciences',
    },
    {
        'Mahidol University In Bangkok',
    },
    {
        'Makerere University College, Kampala, Uganda',
    },
    {
        'Mangalore University, India',
    },
    {
        'Manhattan College',
    },
    {
        'Manhattan School of Music',
    },
    {
        'Manipal University',
    },
    {
        'Mannes College The New School For Music',
    },
    {
        'Mansoura Univeristy',
    },
    {
        'Marche Polytechnic University',
    },
    {
        'Maria Curie-Sklodowska University, Lublin',
    },
    {
        'Mario Negri Institute for Pharmacological Research',
        'Mario Negri Institute for Pharmacological Research, Italy',
    },
    {
        'Marmara University Medical School, Istanbul',
        'Marmara University',
    },
    {
        'Martin Luther University',
    },
    {
        'Maryland Institute College of Art',
    },
    {
        'Masaryk University, Brno, Czech',
    },
    {
        'Mashhad University of Medical Sciences',
    },
    {
        'Massachusetts College of Art and Design',
        'Massachusetts College of Art',
    },
    {
        'Massachusetts General Hospital Institute of Health Professions',
    },
    {
        'Massey University',
    },
    {
        'Max Planck Institute for Biophysical Chemistry',
        'Max Planck Institute of Biochemistry',
    },
    {
        'Max-Planck Institute for Solid State Research',
    },
    {
        'Max-Planck-Institut for Biological Cybernetics',
    },
    {
        'Mayo Clinic College of Medicine, Rochester',
    },
    {
        'McDaniel College',
    },
    {
        'McNeese State University',
    },
    {
        'Medical Academy of Silesia',
        "Silesian School of Medicine",
        "Silesian School of Medicine, Poland",
        "Medical University of Silesia",
    },
    {
        'Medical Academy of Warsaw',
    },
    {
        'Medical College of Pennsylvania',
    },
    {
        'Medical Research Council',
    },
    {
        'Medical School of Athens',
    },
    {
        'Medical University of Lodz',
        "Poland-Academy Med, Lodz",
    },
    {
        'Medical University of Lubeck',
        'Medical University of Luebeck',
    },
    {
        'Medical University of Warsaw',
    },
    {
        'Medizinische Hochschule Hannover',
    },
    {
        'Memphis College of Art',
    },
    {
        'Middle East Technical University',
    },
    {
        'Middlesex University, London',
    },
    {
        'Midwestern Baptist Theological Seminary',
    },
    {
        'Midwestern State University',
    },
    {
        'Mie University Graduate School of Medicine, Mie, Japan',
        'Mie University',
    },
    {
        'Minnesota State University-Mankato',
    },
    {
        'Minsk Medical Institute, Minsk, Belarus',
    },
    {
        "Academy of Science Minsk",
    },
    {
        "Institute Of Radiobiology National Academy Of Sciences, Minsk",
    },
    {
        "Institute of Experimental Botany, Minsk",
    },
    {
        "Institute of Physics Academy of Sciences in Minsk",
    },
    {
        "Minsk Institute of Mathematics",
    },
    {
        "Minsk State Institute for Foreign Languages",
    },
    {
        "Minsk State Linguistic University",
    },
    {
        'Missouri State University',
    },
    {
        'Miyazaki Medical College',
        'Miyazaki Medical College, Japan',
    },
    {
        'Monmouth University',
    },
    {
        'Monterrey Institute of Technology',
    },
    {
        'Morehouse College',
    },
    {
        'Moscow Engineering Physics Institute, Moscow',
    },
    {
        'Moscow Institute Of Industrial',
    },
    {
        'Moscow Institute',
    },
    {
        'Moscow Pedagogical Institute',
        'Moscow State Pedagogical University',
    },
    {
        'Murdoch University',
    },
    {
        'N.N. Blokhin Cancer Research Center, Moscow',
    },
    {
        'NTR University of Health Sciences, Vijayawada Rangaraya Medical College',
    },
    {
        'Nagoya City University Medical School',
        'Nagoya City University',
        'Nagoya University',
    },
    {
        'Nanjing Agricultural University',
    },
    {
        'Nanjing Forestry University',
    },
    {
        'Nanjing University of Science and Technology',
    },
    {
        'Nanjing University',
    },
    {
        'Nankai University',
    },
    {
        'Nantong Medical College',
    },
    {
        'Napier University',
    },
    {
        'Naropa University',
    },
    {
        'National Academy of Sciences of Belarus',
    },
    {
        'National Cardiology Research Center',
    },
    {
        'National Centre for Biological Sciences',
    },
    {
        'National Chemical Laboratory, Pune, India',
    },
    {
        'National Cheng Kung University',
    },
    {
        'National Chiao Tung University',
    },
    {
        'National Dairy Research Institute',
    },
    {
        'National Film and Television School, England',
    },
    {
        'National Institute for Medical Research, London',
    },
    {
        'National Institute of Pharmaceutical Education & Research (NIPER)',
    },
    {
        'National Institute of Pure and Applied Mathematics',
    },
    {
        'National Louis University, Chicago',
    },
    {
        'National Polytechnic Institute of Grenoble',
    },
    {
        'National Tsing-Hua University',
    },
    {
        'National University of San Marcos, Lima',
    },
    {
        'National University of Tucuman',
    },
    {
        'National Yang-Ming University',
    },
    {
        'National-Louis University',
    },
    {
        'Naval War College',
    },
    {
        'New College, Oxford',
        "Exeter College, Oxford",
    },
    {
        'New University of Lisbon',
    },
    {
        'New York Chiropractic College, Old Brookville',
    },
    {
        'New York College of Osteopathic Medicine',
    },
    {
        'New York College of Podiatric Medicine',
    },
    {
        'New York Institute of Technology',
    },
    {
        'Nicholas Copernicus University',
        "Copernicus Medical University",
    },
    {
        'Nihon University School of Medicine, Tokyo',
        'Nihon University',
        "Nippon Medical School",
        "Nippon Medical School, Tokyo, Japan"
        "Tokyo University Medical School",
    },
    {
        'Niigata University',
    },
    {
        'Nizhny Novgorod State University, Nizhny Novgorod',
    },
    {
        'Norman Bethune University of Medical Sciences, China',
        'Norman Bethune University.',
    },
    {
        'North Carolina Central University',
    },
    {
        'North Carolina School of the Arts',
    },
    {
        'North Texas State University, Denton',
    },
    {
        'Northcentral University, Prescott Valley',
    },
    {
        'Northeastern Illinois University',
    },
    {
        'Northeastern Ohio Universities',
    },
    {
        'Northern Jiaotong University',
    },
    {
        'Northern Kentucky University',
    },
    {
        'Northwestern Polytechnical University,China',
    },
    {
        'Norwegian School of Economics & Bus. Admin.',
    },
    {
        'Norwich University',
    },
    {
        'Nova Scotia College of Art And Design',
    },
    {
        'Nova University',
    },
    {
        'Novi Sad, Yugoslavia',
    },
    {
        'Oberlin College And New England Conservatory of Music',
    },
    {
        'Ocean University of Qingdao',
    },
    {
        'Odessa University',
    },
    {
        'Ohio College of Podiatric Medicine, Cleveland',
    },
    {
        'Ohio Northern University',
    },
    {
        'Okayama University',
    },
    {
        'Ontario Veterinary College',
    },
    {
        'Orsay University',
    },
    {
        'Osaka City University',
    },
    {
        'Osaka University',
    },
    {
        'Osmania Medical College, Hyderabad',
        'Osmania University',
    },
    {
        'Oswaldo Cruz Institute',
    },
    {
        'Otis Parsons Art Institute',
    },
    {
        'Otterbein College',
    },
    {
        'Otto-von-Guericke University, Magdeburg',
    },
    {
        'Our Lady of the Lake University, San Antonio',
    },
    {
        'Oxford Polytechnic',
    },
    {
        'Pace University',
    },
    {
        'Pacific Graduate School of Psychology',
    },
    {
        'Pacifica Graduate Institute',
    },
    {
        'Pardee Rand Graduate School',
    },
    {
        'Paris Conservatory',
    },
    {
        'Parsons The New School for Design, New York City',
    },
    {
        'Peabody Conservatory of Music',
    },
    {
        'Pecs University Medical School',
        'Pecs University Medical School, Hungary',
    },
    {
        'Pennsylvania Academy of the Fine Arts',
    },
    {
        'Peruvian University Cayetano Heredia, Lima, Peru',
    },
    {
        'Petersburg Nuclear Physics Institute',
    },
    {
        'Philadelphia College of Pharmacy and Science',
    },
    {
        'Philipps University',
        'Philipps-Universitat Marburg',
    },
    {
        'Physical Research Laboratory, Ahmedabad',
    },
    {
        'Pittsburg State University',
    },
    {
        'Pohang University of Science and Technology',
    },
    {
        'Polish Academy of Sciences',
    },
    {
        'Politecnico di Torino',
        "Politencnico di Torino",
    },
    {
        'Polytechnic Institute of Brooklyn',
    },
    {
        'Polytechnic University of Catalonia',
    },
    {
        'Polytechnic University of Tirana',
    },
    {
        'Polytechnic of Central London',
    },
    {
        'Pondicherry University',
    },
    {
        'Pontifical Catholic University Of Chile',
    },
    {
        'Pontifical Catholic University of Ecuador',
    },
    {
        'Pontifical Institute of Liturgy, Rome, Italy',
    },
    {
        'Portugal-Aveiro University',
    },
    {
        'Poznan University of Medical Sciences',
        "Univeristy of Medical Sciences, Poznan, Poland",
        "Academy of Medicine, Poznan",
        "Poznan Medical Academy",
        "Medical University of Poznan",
    },
    {
        'Pratt Institute, Brooklyn, New York',
    },
    {
        'Prescott College',
    },
    {
        'Qingdao University Medical College, China',
    },
    {
        'Queen Mary and Westfield College',
    },
    {
        'Quinnipiac University',
    },
    {
        'Radford University',
    },
    {
        'Rajiv Gandhi University of Health Sciences',
    },
    {
        'Rand Afrikaans University, Johannesburg, South Africa',
    },
    {
        'Rawalpindi Medical College',
    },
    {
        'Reed College, Portland',
    },
    {
        'Research Institute of Molecular Pathology, Vienna',
    },
    {
        'Rider University',
    },
    {
        'Rio Piedras - University of Puerto Rico',
    },
    {
        'Romanian Academy of Sciences',
    },
    {
        'Ross University School of Medicine',
    },
    {
        'Rowan University',
    },
    {
        'Royal Holloway',
    },
    {
        'Royal Melbourne Institute of Technology',
    },
    {
        'Royal Veterinary College',
    },
    {
        'Ruhr-Universitat Bochum',
    },
    {
        'Ruprecht Karls University',
    },
    {
        'Russian Academy of Medical Science, Moscow',
    },
    {
        'Russian Space Research Institute, Moscow, Russia',
    },
    {
        'Rutgers New Jersey Medical School',
    },
    {
        'Ryerson University',
    },
    {
        'Ryukoku University, Kyoto',
    },
    {
        'SISSA Trieste',
        'SISSA Trieste, Italy',
        'SISSA-International School for Advanced Studies, Trieste',
    },
    {
        'SUNY Cortland',
    },
    {
        'SUNY Empire State College',
    },
    {
        'Saba University School of Medicine, Dutch Carribean',
    },
    {
        'Sackler School of Medicine',
    },
    {
        'Saga School of Medicine',
    },
    {
        "Saint George's Medical School",
    },
    {
        "Saint George's University",
    },
    {
        "Saint John's University",
    },
    {
        "Saint Joseph's University",
    },
    {
        'Saint Louis College of Pharmacy',
    },
    {
        "Saint Mary's University",
    },
    {
        'Saint Petersburg Polytechnical University',
        "St-Petersburg Marine Technical University",
        "St. Petersburg State Technical University, St. Petersburg, Russia",
        "St.Petersburg Technical University"
    },
    {
        'Saint Petersburg State University',
    },
    {
        'Saitama University, Saitama',
    },
    {
        'Salem State University',
    },
    {
        'Samford University',
    },
    {
        'San Diego State University/University of California, San Diego',
    },
    {
        'San Francisco Art Institute',
    },
    {
        'San Francisco Conservatory of Music',
    },
    {
        'San Francisco Theological Seminary',
    },
    {
        "Sant'Anna School of Advanced Studies, Pisa",
    },
    {
        'Sawai Man Singh Medical College',
    },
    {
        'Saybrook University, San Francisco',
    },
    {
        'School Of The Museum Of Fine Arts, Boston, Ma',
    },
    {
        'School of Advanced Studies in Social Sciences, Paris',
    },
    {
        'School of Dentistry of the National and Kapodistrian University, Athens',
    },
    {
        'School of Medicine at Charles University',
    },
    {
        'School of Medicine, University of Buenos Aires, Argentina',
    },
    {
        'School of Visual Arts, New York',
    },
    {
        'Science Center, State University of New York, Brooklyn',
    },
    {
        'Science University of Tokyo',
        "Tokyo University of Science, Japan"
    },
    {
        'Scuola Internazionale Superiore Di Studi Avanzati Di Trieste',
    },
    {
        "Scuola Superiore Sant'Anna, Pisa",
    },
    {
        'Sechenov Institute of Evolutionary Physiology and Biochemistry',
    },
    {
        'Second Military Medical University',
    },
    {
        'Seoul National University',
    },
    {
        'Seth G.S. Medical College',
    },
    {
        'Shandong University',
        'Shandong University, Jinan',
    },
    {
        'Shanghai 2Nd Medical University, China',
        'Shanghai Second Medical University',
    },
    {
        'Shanghai Institute for Biological Sciences',
        'Shanghai Institutes for Biological Sciences',
    },
    {
        'Shanghai Institute of Biochemistry, Shanghai',
    },
    {
        'Shanghai Institute of Cell Biology',
    },
    {
        'Shanghai Institute of Ceramics',
    },
    {
        'Shanghai Institute of Organic Chemistry',
    },
    {
        'Shanghai Institute of Physiology',
        'Shanghai Institute of Physiology, Chinese Academy of Sciences',
    },
    {
        'Shanghai Institute',
    },
    {
        'Shanghai Jiao Tong University School of Medicine, China',
        'Shanghai Jiao Tong University',
    },
    {
        'Shanghai Medical School , Shanghai, China',
    },
    {
        'Shanghai University of Science and Technology, China',
    },
    {
        'Sharif University of Technology',
    },
    {
        'Shenyang Pharmaceutical University, China',
    },
    {
        'Shiga University of Medical Science, Shiga, Japan',
    },
    {
        'Shinshu University',
    },
    {
        'Shiraz University School of Medicine, Shiraz',
        'Shiraz University',
    },
    {
        'Shirshov Institute of Oceanology',
    },
    {
        'Showa University',
    },
    {
        'Siberian State Medical University',
        "Siberian State Medical University, Tomsk",
    },
    {
        'Sichuan University, China',
    },
    {
        'Silesian University of Technology',
        "Technical University of Silesia, Poland",
    },
    {
        'Simon Fraser University',
    },
    {
        'Siriraj Medical School, Mahidol University, Bangkok',
    },
    {
        'Sissa-Isas International School For Advanced Studies',
    },
    {
        'Slovak Academy Of Sciences',
        'Slovak Academy of Sciences, Bratislava',
    },
    {
        'Slovak Technical University',
    },
    {
        'Sobolev Institute of Mathematics, Novosibirsk',
    },
    {
        'Sogang University',
    },
    {
        'Sonoma State University',
    },
    {
        'Soochow University, China',
    },
    {
        'Sophia University',
    },
    {
        'South China University Of Tropical Agriculture, Hainan',
    },
    {
        'South China University of Technology, China',
    },
    {
        'South Texas College of Law',
    },
    {
        'Southeast University, China',
    },
    {
        'Southeastern University',
    },
    {
        'Southern California Institute of Architecture',
    },
    {
        'Southern College of Optometry',
    },
    {
        'Southern Illinois University School of Medicine',
    },
    {
        'Southern Illinois University, Springfield',
    },
    {
        'Southern New Hampshire University',
    },
    {
        'Southern Oregon University',
    },
    {
        'Southern University at New Orleans',
    },
    {
        'Southwest Texas State University',
    },
    {
        'Southwestern University School of Law',
    },
    {
        'Southwestern University',
    },
    {
        'Spartan Health Sciences University',
    },
    {
        'Sree Chitra Tirunal Institute of Medical Sciences and Technology, India',
        "Sree Chitra Tirunal Institute for Medical Sciences & Technology",
        "Sree Chitra Tirunal Institute of Medical Sciences and Technology"
    },
    {
        'Sri Krishna Devaraya University',
        "Sri Krishna Devaraya University, India",
        "Sri Krishnadevaray University, India"
    },
    {
        'Sri Ramachandra Medical College And Research Institute, Chennai',
    },
    {
        'St John Fisher College',
    },
    {
        "St. Bartholomew's Hospital College",
    },
    {
        "St. John Fisher College, Rochester, New York",
    },
    {
        "St. John's Medical College, Bangalore, India",
    },
    {
        "St. Matthew's University",
    },
    {
        "St. Petersburg State Conservatory, Russia",
    },
    {
        "St. Thomas Institute",
    },
    {
        "St. Thomas's Hospital Medical School",
    },
    {
        'St.Johns Medical College',
    },
    {
        'State University of Haiti',
    },
    {
        'State University of New York Brooklyn',
    },
    {
        'State University of New York Health Science Center at Brooklyn',
    },
    {
        'State University of New York, Mount Sinai',
    },
    {
        'State University of New York, Purchase',
    },
    {
        'Stetson University College of Law',
    },
    {
        'Stetson University',
    },
    {
        'Stockholm School of Economics',
    },
    {
        'Suez Canal University',
    },
    {
        'Sungkyunkwan University',
    },
    {
        'Suzhou Medical College, China',
    },
    {
        'Swinburne University of Technology',
    },
    {
        'Swiss Federal Institute of Technology',
    },
    {
        'Swiss Institute for Experimental Cancer Research',
    },
    {
        'Szent Istvan University',
        'Szent Istvan University, Hungary',
    },
    {
        'Taipei Medical University',
    },
    {
        'Taishan Medical College, China',
    },
    {
        'Tamil Nadu Dr. MGR Medical University, Chennai, India',
    },
    {
        'Tanta University',
    },
    {
        'Tarbiat Modarres University',
        'Tarbiat Modarres University, Tehran',
    },
    {
        'Tashkent State University',
    },
    {
        'Tata Institute of Fundamental Research,the',
    },
    {
        'Tbilisi State Medical University, Tbilisi, Georgia',
        'Tbilisi State University',
    },
    {
        'Technical University Braunschweig',
    },
    {
        'Technical University in Graz',
    },
    {
        'Technical University in Wroclaw',
    },
    {
        'Technical University of Gdansk',
        "Polytechnic Institute of Gdansk",
    },
    {
        'Technical University, Sofia',
    },
    {
        'Technische  Universitaet Darmstadt',
    },
    {
        'Technische Universitaet Wien',
    },
    {
        'Tel-Aviv University',
    },
    {
        'Texas A&M University, Galveston',
    },
    {
        'Texas College of Osteopathic Medicine',
    },
    {
        'Texas Wesleyan University',
    },
    {
        'Thanjavur Medical College, Thanjamur',
    },
    {
        'The Flinders University of South Australia',
    },
    {
        'The Middlesex Hospital Medical School',
    },
    {
        'The University of Cantabria, Spain',
    },
    {
        'The University of Kansasin',
    },
    {
        "Tver State Medical Academy",
        "Tver State Medical Academy, Russia",
    },
    {
        "Tver State University, Tver",
        "Tver State University, Tver, Russia",
    },
    {
        'The University of Windsor',
    },
    {
        'Third Military Medical University, China',
    },
    {
        'Thomas Cooley Law School',
    },
    {
        'Tianjin Medical University',
    },
    {
        'Tianjin University',
    },
    {
        'Tilburg University',
    },
    {
        'Tirunelveli Medical College, Madurai University, India',
    },
    {
        'Tishreen University',
    },
    {
        'Tohoku University',
    },
    {
        "Toho Gakuen School of Music",
        "Toho Gakuen School of Music, Japan",
        "Toho School of Music",
    },
    {
        'Tokai University',
        'Tokai University, Japan',
    },
    {
        'Tokyo Medical and Dental University',
    },
    {
        'Tokyo Metropolitan University',
    },
    {
        'Tokyo University of Agriculture and Technology',
    },
    {
        'Tongji Medical University',
    },
    {
        'Topiwala National Medical College, Mumbai',
    },
    {
        'Touro College of Law',
    },
    {
        'Touro University International',
    },
    {
        'Towson University',
    },
    {
        'Toyama Medical and Pharmaceutical University',
    },
    {
        'Toyota Technological Institute at Chicago',
    },
    {
        'Trevecca Nazarene University',
    },
    {
        'Trinity College, Hartford',
    },
    {
        'Trinity Evangelical Divinity School',
    },
    {
        'Tsinghua University',
    },
    {
        'Tuskegee University',
    },
    {
        'UNIVERSITY OF WISCONSIN - STEVENS POINT',
    },
    {
        'US International University',
    },
    {
        'Ukrainian Academy of Sciences',
    },
    {
        'Ultrecht University',
    },
    {
        'United States Sports Academy',
        "United States Sports Academy Daphne, Alabama",
    },
    {
        'Univeristy of Aachen',
    },
    {
        'Universidad Anahuac',
    },
    {
        'Universidad Austral de Chile',
    },
    {
        'Universidad Autonoma De Mexico (UNAM)',
    },
    {
        'Universidad Autonoma Metropolitana',
    },
    {
        'Universidad Autonoma de Guadalajara',
    },
    {
        'Universidad Autonoma de Madrid',
    },
    {
        'Universidad Carlos III de Madrid',
    },
    {
        'Universidad Catolica de Santiago de Guayaquil',
    },
    {
        'Universidad Del Salvador',
    },
    {
        'Universidad Del Zulia',
    },
    {
        'Universidad El Bosque',
    },
    {
        'Universidad Industrial de Santander',
    },
    {
        'Universidad La Salle',
    },
    {
        'Universidad Michoacana San Nicolas de Hidalgo',
    },
    {
        'Universidad Nacional Autonoma de Mexico',
    },
    {
        'Universidad Nacional de EducaciÜn a Distancia, Madrid, Spain',
    },
    {
        'Universidad Politecnica de Catalunya (UPC)',
    },
    {
        'Universidad Politecnica de Madrid',
        'Universidad Politecnica de Madrid, Spain',
    },
    {
        'Universidad de Alcala de Henares, Madrid',
    },
    {
        'Universidad de Barcelona, Barcelona',
        "Universidad de Barcelona,",
        "Universitat de Barcelona",
    },
    {
        'Universidad de Cantabria',
        'Universidad de Cantabria, Spain',
    },
    {
        'Universidad de Cuyo and Instituto Balseiro',
    },
    {
        'Universidad de Guadalajara',
    },
    {
        'Universidad de LaSalle, Bogota, Colombia',
    },
    {
        'Universidad de Monterrey',
    },
    {
        'Universidad de Murcia',
    },
    {
        'Universidad de Sevilla (Spain)',
    },
    {
        'Universidad del Norte',
    },
    {
        'Universidad del Rosario, Bogota',
    },
    {
        'Universidad del Valle, Cali',
    },
    {
        'Universidade Estadual de Campinas, Sao Paulo',
    },
    {
        'Universita Degli Studi Di Pavia',
    },
    {
        'Universita Degli Studi Di Roma La Sapienza',
    },
    {
        'Universita Vita-Salute San Raffaele, Milano',
    },
    {
        'Universita degli Studi Roma Tre',
    },
    {
        'Universitat Passau',
    },
    {
        'Universitat Ramon Llull, Esade',
    },
    {
        'Universitat Rovira i Virgili',
    },
    {
        'Universite Catholique de Louvain',
    },
    {
        'Universite Laval',
    },
    {
        'Universite Libre de Bruxelles',
    },
    {
        'Universite Lumiere, Lyon II',
        "Universite de Lyon II",
    },
    {
        'Universite Victor Segalen',
    },
    {
        'Universite de Limoges',
        'Universite de Limoges, France',
    },
    {
        'Universiti Putra Malaysia',
    },
    {
        'University Autonoma De Nuevo Leon',
    },
    {
        'University Blaise Pascal',
    },
    {
        'University College Of Medical Sciences',
    },
    {
        'University Halle-Wittenberg',
    },
    {
        'University Hospital of Geneva and University of Savoy',
    },
    {
        'University Medical School of Debrecen',
        "Debrecen University Medical School, Debrecen",
    },
    {
        'University Of Al Fateh',
    },
    {
        'University Of The East Ramon Magsaysay Memorial Medical center',
    },
    {
        'University Of Veterinary Medicine, Hannover, Germany',
    },
    {
        'University Of Veterinary Medicine, Vienna, Austria',
    },
    {
        'University Paul Cezanne',
    },
    {
        'University Politehnica Bucharest',
    },
    {
        'University degli Studi of Napoli Federico',
    },
    {
        'University do Porto',
    },
    {
        'University in Trento',
    },
    {
        'University of A Coruna',
    },
    {
        'University of Aberdeen',
    },
    {
        'University of Aix-Marseille, France',
    },
    {
        'University of Alaska Anchorage',
    },
    {
        'University of Alaska Southeast',
    },
    {
        'University of Alcala',
    },
    {
        'University of Algiers',
    },
    {
        'University of Amsterdam',
    },
    {
        'University of Ancona',
    },
    {
        'University of Angers',
    },
    {
        'University of Antwerp',
    },
    {
        'University of Baghdad',
    },
    {
        'University of Baria',
    },
    {
        'University of Bayreuth',
    },
    {
        'University of Benin',
    },
    {
        'University of Bergen',
    },
    {
        'University of Bologna',
    },
    {
        'University of Bordeaux',
    },
    {
        'University of Braunschweig,the',
    },
    {
        'University of Bremen',
    },
    {
        'University of Bridgeport',
    },
    {
        'University of Bucharest',
    },
    {
        'University of Burgundy',
    },
    {
        'University of Caen',
    },
    {
        'University of Calabria',
    },
    {
        'University of Calicut',
    },
    {
        'University of California, Hastings College of Law',
    },
    {
        'University of Camerino',
    },
    {
        'University of Cantabria',
    },
    {
        'University of Cape Town',
    },
    {
        'University of Cartagena Medical School, Cartagena, Colombia',
    },
    {
        'University of Catalonia',
    },
    {
        'University of Central Lancashire',
    },
    {
        'University of Chile',
    },
    {
        'University of Clermont-Ferrand',
    },
    {
        'University of Concepcion',
    },
    {
        'University of Craiova',
    },
    {
        'University of Crete',
    },
    {
        'University of Debrecen',
    },
    {
        'University of Denmark, Copenhagen-Lyngby',
    },
    {
        'University of Deusto',
    },
    {
        'University of Dijon',
        'University of Dijon, France',
    },
    {
        'University of Dortmund',
    },
    {
        'University of Dundee',
    },
    {
        'University of Dusseldorf',
    },
    {
        'University of East Anglia',
    },
    {
        'University of East London',
    },
    {
        'University of Erlangen',
    },
    {
        'University of Ferrara',
        'University of Ferrara, Italy',
    },
    {
        'University of Findlay',
    },
    {
        'University of Florence',
        'Universita Degli Studi Di Firenze',
        "University of Firenze",
    },
    {
        'University of Frankfurt',
    },
    {
        'University of Freiburg',
        'University of Fribourg',
    },
    {
        'University of Gdansk',
    },
    {
        'University of Giessen',
    },
    {
        'University of Girona',
    },
    {
        'University of Gloucestershire, Cheltenham',
    },
    {
        'University of Greenwich',
    },
    {
        'University of Guelph',
    },
    {
        'University of Halle',
    },
    {
        'University of Hannover',
    },
    {
        'University of Haute Alsace',
    },
    {
        'University of Havana',
    },
    {
        'University of Health Sciences Christian Medical College, Ludhiana',
    },
    {
        'University of Health Sciences College of Osteopathy',
    },
    {
        'University of Hyderabad',
    },
    {
        'University of Ia Roy J & L Carver Com',
    },
    {
        'University of Iceland Reykjavik',
    },
    {
        'University of Illinois at Springfield',
    },
    {
        'University of Innsbruck in Austria',
    },
    {
        'University of Jena',
    },
    {
        'University of Jussieu, Paris',
    },
    {
        'University of Kalyani',
    },
    {
        'University of Karachi',
    },
    {
        'University of Kashmir',
    },
    {
        'University of Kassel',
    },
    {
        'University of Kent',
    },
    {
        'University of Kerala',
    },
    {
        'University of Kiel',
    },
    {
        'University of Koeln',
    },
    {
        'University of Konstanz',
    },
    {
        'University of Kuopio',
    },
    {
        'University of KwaZulu-Natal',
    },
    {
        "University of L'Aquila",
    },
    {
        'University of La Laguna',
    },
    {
        'University of La Verne',
    },
    {
        'University of Lausanne',
    },
    {
        'University of Leicester',
    },
    {
        'University of Leiden',
    },
    {
        'University of Leipzig',
    },
    {
        'University of Leuven',
    },
    {
        'University of Lille',
    },
    {
        'University of Ljubljana',
    },
    {
        'University of Lorraine',
    },
    {
        'University of Louvain',
    },
    {
        'University of Lowell',
    },
    {
        'University of Luebeck',
    },
    {
        'University of Lugano',
    },
    {
        'University of Lyon',
    },
    {
        'University of Madrid',
        'University of Madrid, Spain',
    },
    {
        'University of Malaga, Spain',
    },
    {
        'University of Malta Medical School',
    },
    {
        'University of Manchester Institute of Science and Technology',
    },
    {
        'University of Manitoba',
    },
    {
        'University of Marburg',
    },
    {
        'University of Mary Hardin-Baylor',
    },
    {
        'University of Mary',
    },
    {
        'University of Massachusetts- Worcester',
    },
    {
        'University of Medical Sciences (Tehran, Iran)',
    },
    {
        'University of Medicine and Pharmacy',
    },
    {
        'University of Messina',
    },
    {
        'University of Metz',
    },
    {
        'University of Mexico, Albuquerque',
    },
    {
        'University of Minho',
        'University of Minho, Portugal',
    },
    {
        'University of Mining and Metallurgy, Cracow',
    },
    {
        'University of Monterrey',
    },
    {
        'University of Montpellier II',
    },
    {
        'University of Montpellier',
    },
    {
        'University of Munster',
    },
    {
        'University of Murcia',
    },
    {
        'University of Mysore',
    },
    {
        'University of Nantes',
    },
    {
        'University of Natural Resources and Life Sciences, Vienna (BOKU)',
    },
    {
        'University of Natural Resurces and Applied Life Sciences',
    },
    {
        'University of Navarra',
    },
    {
        'University of Nebraska Kearney',
    },
    {
        'University of New Haven, Connecticut',
    },
    {
        'University of Newcastle upon Tyne',
    },
    {
        'University of Nice',
    },
    {
        'University of North Florida',
    },
    {
        'University of Novi Sad',
    },
    {
        'University of Oslo',
        "Oslo University",
        "Oslo University, Norway",
    },
    {
        'University of Osnabruck',
    },
    {
        'University of Otago',
    },
    {
        'University of Oulu',
    },
    {
        'University of Oviedo',
    },
    {
        'University of Paderborn',
    },
    {
        'University of Padova',
        "Universit   Degli Studi Di Padova",
    },
    {
        'University of Padua',
    },
    {
        'University of Paisley',
    },
    {
        'University of Palermo School of Medicine',
    },
    {
        'University of Palermo',
        "Universita degli Studi di Palermo",
        'University of Palermo, Italy',
    },
    {
        'University of Paris XII (Val de Marne)',
    },
    {
        'University of Parma (Italy)',
    },
    {
        'University of Patras',
        'University of Patras, Greece',
    },
    {
        'University of Pecs',
        'University of Pecs, Hungary',
    },
    {
        'University of Perpignan',
    },
    {
        'University of Perugia',
    },
    {
        'University of Picardie, Amiens, France',
    },
    {
        'University of Pisa',
    },
    {
        'University of Poitiers',
    },
    {
        'University of Port Harcourt',
    },
    {
        'University of Portland, Portland, Oregon',
    },
    {
        'University of Portsmouth',
    },
    {
        'University of Potsdam',
    },
    {
        'University of Poznan',
    },
    {
        'University of Prince Edward Island',
    },
    {
        'University of Provence',
    },
    {
        'University of Pune',
    },
    {
        'University of Quebec',
    },
    {
        'University of Rajasthan (India)',
    },
    {
        'University of Reading',
    },
    {
        'University of Redlands',
    },
    {
        'University of Regensburg',
    },
    {
        'University of Reims Champagne, Ardennes',
    },
    {
        'University of Richmond',
    },
    {
        'University of Roehampton',
    },
    {
        'University of Roma',
    },
    {
        'University of Rome Tor Vergata',
    },
    {
        'University of Roorkee',
    },
    {
        'University of Rostock',
    },
    {
        'University of Salerno',
        'University of Salerno, Italy',
    },
    {
        'University of Salford, England',
    },
    {
        'University of Salzburg',
    },
    {
        'University of San Carlos in Guatemala City',
    },
    {
        'University of San Luis,the',
    },
    {
        'University of Santiago de Compostela',
    },
    {
        'University of Santo Amaro',
    },
    {
        'University of Sarajevo',
    },
    {
        'University of Sarasota',
    },
    {
        'University of Saskatchewan',
    },
    {
        'University of Science and Technologies, Lille',
    },
    {
        'University of Scranton',
    },
    {
        'University of Sherbrooke, Quebec',
    },
    {
        'University of Siegen',
    },
    {
        'University of Siena',
    },
    {
        'University of South Australia',
    },
    {
        'University of South Wales',
    },
    {
        'University of Southwestern Louisiana',
    },
    {
        'University of St. Augustine',
    },
    {
        'University of St. Francis',
    },
    {
        'University of St. Gallen',
    },
    {
        "University of St. Michael's College",
    },
    {
        'University of St. Thomas',
    },
    {
        'University of Stirling',
    },
    {
        'University of Strasbourg',
    },
    {
        'University of Stuttgart',
    },
    {
        'University of Sunderland',
    },
    {
        'University of Tampere',
    },
    {
        'University of Tartu',
    },
    {
        'University of Tasmania',
    },
    {
        'University of Technology in Vienna',
    },
    {
        'University of Teheran',
    },
    {
        'University of Thessaloniki',
    },
    {
        'University of Tokyo',
    },
    {
        'University of Torino',
        "Universita degli Studi di Torino, Italy",
    },
    {
        'University of Toulouse',
    },
    {
        'University of Trier',
    },
    {
        'University of Trieste',
    },
    {
        'University of Tsukuba',
    },
    {
        'University of Turin',
    },
    {
        'University of Twente, Enschede, The Netherlands',
    },
    {
        'University of Udine, Italy',
        "University of Udine",
        "University of Udine, Udine",
    },
    {
        'University of Uruguay, Montevideo, Uruguay',
    },
    {
        'University of Van Amsterdam',
    },
    {
        'University of Veszprem',
        'University of Veszprem, Hungary',
    },
    {
        'University of Victoria',
    },
    {
        'University of Vigo',
    },
    {
        'University of Warmia and Mazury, Olsztyn',
    },
    {
        'University of West Georgia',
    },
    {
        'University of West London, UK',
    },
    {
        'University of Western Brittany',
    },
    {
        'University of Westminster',
    },
    {
        'University of Wisconsin, Eau Claire',
    },
    {
        'University of Wisconsin, La Crosse',
    },
    {
        'University of Wisconsin, Oshkosh',
    },
    {
        'University of Witten',
    },
    {
        'University of Wolverhampton',
    },
    {
        'University of Zulia',
    },
    {
        'University of Zurich',
    },
    {
        'University of the Arts, Philadelphia',
    },
    {
        'University of the Balearic Islands',
    },
    {
        'University of the East',
    },
    {
        'University of the Mediterranean',
    },
    {
        'University of the Orange Free State, Bloemfontein',
    },
    {
        'University of the Republic of Uruguay Medical School, Montevideo',
        'University of the Republic, Montevideo, Uruguay',
    },
    {
        'University of the Rockies',
    },
    {
        'University of the West Indies, St. Augustine, Trinidad',
    },
    {
        'Università Cattolica Del Sacro Cuore',
    },
    {
        "Universitá G. d'Annunzio, Chieti-Pescara, Italy",
    },
    {
        'Unversity of Bordeaux I',
    },
    {
        'Unversity of Bordeaux II',
    },
    {
        'Ural State University',
    },
    {
        'Utkal University',
    },
    {
        'VNII Genetika, Moscow',
    },
    {
        'Vandercook College, Chicago',
    },
    {
        'Vermont Law School',
    },
    {
        'Victor Babes University of Medicine and Pharmacy, Timisoara',
        "University of Medicine, Timisoara",
        "Medical Univ of Timisoara",
    },
    {
        'Victoria University of Wellington',
    },
    {
        'Vienna Academy of Music and Dramatic Arts',
    },
    {
        'Vienna University of Technology in Austria',
    },
    {
        'Virginia Consortium for Professional Psychology',
    },
    {
        'Voeikov Main Geophysical Observatory, Leningrad',
    },
    {
        'Vollum Institute',
    },
    {
        'Warren Wilson College, North Carolina',
    },
    {
        'Warsaw Institute of Technology',
    },
    {
        'Warsaw School of Economics',
    },
    {
        'Waseda University',
        'Waseda University, Tokyo',
    },
    {
        'Weber State University',
    },
    {
        'Weifang Medical College, China',
    },
    {
        'West Chester University',
    },
    {
        'West China University of Medical Sciences, China',
    },
    {
        'West China University',
    },
    {
        'West Virginia School of Osteopathic Medicine',
    },
    {
        "Western Governor's University",
    },
    {
        'Western Kentucky University',
    },
    {
        'Western New England University',
    },
    {
        'Western University of Health Sciences',
    },
    {
        'Whittier College',
    },
    {
        'Whitworth College',
    },
    {
        'Wilfrid Laurier University',
    },
    {
        'Wilkes University, Wilkes-Barre',
    },
    {
        'William Carey College, Hattiesburg',
    },
    {
        'William Carey University',
    },
    {
        'William Mitchell College of Law',
    },
    {
        'William Paterson University of New Jersey',
    },
    {
        'William S. Richardson School of Law',
    },
    {
        'Wilmington University',
    },
    {
        "Woman's Medical College of Pennsylvania",
    },
    {
        'Wroclaw Medical University',
    },
    {
        'Wuhan University of  Technology',
    },
    {
        'Wuhan University',
    },
    {
        'Wuppertal University',
        "University of Wuppertal, Germany",
        "Bergische University of Wuppertal",
    },
    {
        'Xavier University',
    },
    {
        'Xavier University, Campus Unknown',
    },
    {
        'Xavier University, Cincinnati',
    },
    {
        'Xiamen University',
    },
    {
        'Xian Jiaotong University',
    },
    {
        'Xian Medical University',
    },
    {
        'Xiang-Ya Medical College',
    },
    {
        'Xidian University',
    },
    {
        'Xuzhou Medical College',
    },
    {
        'Yamagata University',
        'Yamagata University, Japan',
    },
    {
        'Yamanashi Medical University, Yamanashi, Japan',
    },
    {
        'Yerevan Physics Institute',
        'Yerevan Physics Institute, Armenia',
    },
    {
        'Yerevan State University',
    },
    {
        'Yokohama City University',
    },
    {
        'Yokohama National University',
    },
    {
        'Zhejiang Medical University',
    },
    {
        'Zhejiang University',
    },
    {
        'École Polytechnique de Montréal, Montreal, Canada',
    },
    {
        "the University of Zambia",
    },
    {
        "the Sandberg Institute (Amsterdam)",
    },
    {
        "Interuniversity Center for Astronomy and Astrophysics",
        "Interuniversity Center for Astronomy and Astrophysics, India",
    },
    {
        "William Jewell College",
    },
    {
        "Moscow Semashko'S Medical Institute",
        "Moscow Semashko'S Medical Institute, Russia",
    },
    {
        "Institute of Steel and Alloys, Moscow",
        "Moscow Institute of Steel and Alloys",
    },
    {
        "Institute of Sociology Academy of Sciences",
        "Institute of Sociology Academy of Sciences, Russia",
    },
    {
        "Blood Center of Wisconsin, Milwaukee",
        "Bloodcenter of Wisconsin, Inc.",
    },
    {
        "Federal University of Goias School of Medicine",
        "Federal University of Goias School of Medicine, Brazil",
    },
    {
        "Federal University of Pernambuco, Pernambuco, Brazil",
        "University Federal of Pernambuco",
    },
    {
        "First Medical School University of Naples",
        "First Medical School University of Naples, Italy",
    },
    {
        "Fritz Haber Institute of the Max Planck Society",
        "Fritz-Haber-Institute",
    },
    {
        "Fundacao Getulio Vargas",
        "Fundação Getúlio Vargas-EAESP",
    },
    {
        "GEMPPM Laboratory in Villeurbanne",
        "GEMPPM Laboratory in Villeurbanne, France",
    },
    {
        "Gembloux Agricultural University",
        "Faculte des Sciences Agronomiques  letat, Gembloux, Belgium,",
        "Gembloux University",
    },
    {
        "Institute for Superhard Materials",
        "Institute for Superhard Materials, Russia",
    },
    {
        "Institute of A.M. Prokhorov",
        "Institute of A.M. Prokhorov, Russia",
    },
    {
        "Institute of Macromolecular Chemistry, Prague",
        "Institute of Chemical Macromolecular Chemistry, Prague",
    },
    {
        "Kendall College",
    },
    {
        "Kiit School Of Management",
    },
    {
        "Kottayam Medical College",
    },
    {
        "Kurnakov's Institute",
    },
    {
        "University of Barri",
    },
    {
        "Institute of Nuclear Physics, Jagiellonian University, Krakow, Poland",
        "Institute of Nuclear Physics, Krakow, Poland",
    },
    {
        "University of Tromso",
        "University of Tromsoe",
    },
    {
        "University of Ulsan",
    },
    {
        "Polytechnic Institute of Timisoara",
        "Polytechnic Institute of Timisoara, Romania",
    },
    {
        "University of West Timisoara",
        "University of West Timisoara, Romania",
    },
    {
        "Berlage Institute",
    },
    {
        "Birzeit University",
    },
    {
        "Bronxville",
    },
    {
        "World Information Distributed University, Brussels",
    },
    {
        "Urmia University, Iran",
    },
    {
        "Utsunomiya University", "Utsunomiya University, Japan",
    },
    {
        "Ivanova state university",
    },
    {
        "Jinan University",
        "Jinan University, China",
    },
    {
        "Kathmandu University", "Kathmandu University, Nepal",
    },
    {
        "Westfaelishe Wilhelms University",
        "Westfalische–Wilhelms-University, Munster",
    },
    {
        "University of Vecha, Germany", "University of Vechta",
    },
    {
        "Prince of Songkla University",
        "Prince of Songkla University, Thailand",
    },
    {
        "University of Agriculture, Faisalabad",
    },
    {
        "University of Beirut", "University of Beirut, Lebanon",
    },
    {
        "Beirut Arab University",
    },
    {
        "Orient-Institut Beirut",
    },
    {
        "Ottawa University, Kansas City",
    },
    {
        "Shiga University",
    },
    {
        "Shiga University of Medical Science, Japan",
    },
    {
        "Sidi Mohamed Ben Abdellah University",
    },
    {
        "Kirensky Institute of Physics",
        "Kirensky Institute of Physics, Krasnoyarsk",
    },
    {
        "China-Kunming",
    },
    {
        "Kunming Medical College, China",
    },
    {
        "Kyushu Dental College",
        "Kyushu Dental School",
    },
    {
        "College of New Rochelle",
    },
    {
        "Concordia College",
    },
    {
        "Concordia College, Moorhead",
    },
    {
        "Concordia College, St Paul",
    },
    {
        "Conservatoire Royal de Bruxelles",
        "Conservatoire Royal de Bruxelles, Belgium",
    },
    {
        "Davao Medical School,",
    },
    {
        "Renmin University of China",
    },
    {
        "Riga Technical University",
        "Riga Technical University, Latvia",
        "Riga TechnicaláUniversity",
    },
    {
        "Ripon Law School",
    },
    {
        "Ritsumeikan University",
        "Ritsumeikan University, Japan",
    },
    {
        "Rijksuniversiteit te Gent",
        "Rijksuniversiteit te Gent, Belgium",
    },
    {
        "Rockefeller Institute",
    },
    {
        "Rosemont College",
    },
    {
        "Roskilde University",
        "Roskilde University, Denmark",
    },
    {
        "Rani Durgavati Vishwavidhyalaya", "Rani Durgavati Vishwavidhyalaya, India",
    },
    {
        "Colby College",
        "COLBY COLLEGE",
    },
    {
        "Donetsk National Technical University",
        "National Technical University, Donetsk",
    },
    {
        "Donetsk State University, Donetsk",
    },
    {
        "VizirLabs",
    },
    {
        "Tottori University",
        "Tottori University, Japan",
        "Tottori University, Yonago",
    },
    {
        "WHU Otto Beisheim School of Management",
        "WHU Otto Beisheim School of Management, Germany",
    },
    {
        "Wannan Medical College, China",
    },
    {
        "Wakayama Medical University",
    },
    {
        "Walla Walla College",
    },
    {
        "Vinayaka Missions University, India",
    },
    {
        "BECKMAN RESEARCH INSTITUTE",
        "Beckman Research Institute",
        "Beckman Research Institute of City of Hope",
    },
    {
        "Boreskov Institute of Catalysis",
        "Boreskov Institute of Catalysis, Russia",
    },
    {
        "Bethany College",
    },
    {
        "Bhabha Atomic Research Center",
        "Bhabha Atomic Research Center, India",
    },
    {
        "Bharathidasan University",
        "Bharathidasan University, India",
    },
    {
        "Bihar University",
    },
    {
        "Binzhou Medical College",
    },
    {
        "Benjamin N. Cardozo",
    },
    {
        "Bastyr University",
    },
    {
        "BARNARD COLLEGE",
        "Barnard College",
    },
    {
        "BERKELEE COLLEGE",
        "Boston Conservatory of Music",
    },
    {
        "Baldwin Wallace University",
    },
    {
        "Bamberg ?University",
        "Bamberg ?University, Germany",
        "the University of Bamberg",
        "University Bamberg in Germany",
    },
    {
        "Baker College",
    },
    {
        "B J College, Pune", "Byramjee Jeejeebhoy Medical College, Pune, India",
    },
    {
        "Audrey Cohen College, New York",
    },
    {
        "Geneva College",
    },
    {
        "Roger Williams University",
    },
    {
        "George Williams College",
    },
    {
        "University of Fukui School of Medicine",
        "University of Fukui School of Medicine, Japan",
        "Fukui Medical University of Japan",
    },
    {
        "University of Ghana Medical School",
    },
    {
        "Technical University of Valencia",
        "Polytechnic University of Valencia",
    },
    {
        "Silesian University",
        "Silesian University, Poland",
    },
    {
        "University of Osaka Prefecture",
        "University of Osaka Prefecture, Japan",
        "Osaka Prefecture University",
    },
    {
        "Osaka Dental University",
    },
    {
        "Virology Osaka University Medical School",
        "Virology Osaka University Medical School, Japan",
    },
    {
        "Vijayanagar Institute of Medical Sciences(VIMS)",
        "Vijayanagar Institute of Medical Sciences(VIMS), India",
    },
    {
        "Vikram University",
        "Vikram University, India",
    },
    {
        "University of Southern Nevada",
    },
    {
        "Mickiewicz University",
        "Mickiewicz University, Poland",
        "Poland-Adam Mickiewicz Univ",
        "University A.Mickiewicz in Pozna?",
        "Chemistry Adam Mickiewicz University, Poznan, Poland",
        "A. Mickiewicz University",
    },
    {
        "A.R.C. Poultry Research Center, Roslin, Scotland",
    },
    {
        "ALLEGHENY COLLEGE",
        "Allegheny College",
    },
    {
        "Abubakar Tafawa Balewa University",
    },
    {
        "Man-Made Fibers Institute at Lodz Polytechnic",
        "Man-Made Fibers Institute at Lodz Polytechnic, Poland",
    },
    {
        "Lodz Academy of Music",
        "Lodz Academy of Music, Poland",
    },
    {
        "Lodz Film School of Poland",
        "Lodz Film School of Poland, Poland",
    },
    {
        "Lithuanian Academy of Physical Education",
    },
    {
        "Lipscomb University",
    },
    {
        "Rollins College, Winter Park, FL",
    },
    {
        "Rosary College",
    },
    {
        "State University of Chernovtsy, Ukraine",
        "University of Chernovtsy",
    },
    {
        "Staatliche Akademie der Bildenden Künste, Karlsruhe, Germany.",
        "State Academy Of Fine Arts, Karlsruhe, Germany",
    },
    {
        "State University of Karlsruhe Gestaltung",
    },
    {
        "Domus Academy, Milan, Italy",
    },
    {
        "Drury University",
    },
    {
        "Dokuz Eylul University",
        "Dokuz Eylul University, Turkey",
    },
    {
        "Davenport University",
    },
    {
        "Daemen College",
    },
    {
        "Condensed Matter Physics Indian Institute of Technology",
    },
    {
        "Columbus College of Art & Design",
    },
    {
        "Columbus State University",
    },
    {
        "Samara State Technical University",
        "Samara State Technical University, Russia",
    },
    {
        "American College of Veterinary",
    },
    {
        "Alexandru Ioan Cuza University",
        "Alexandru Ioan Cuza University, Romania",
    },
    {
        "Agricultural University of Poznan",
    },
    {
        "Poznan Academy of Fine Arts",
    },
    {
        "Poznan University of Life Sciences",
        "Poznan University of Life Sciences, Poland",
    },
    {
        "Academy of Science Institute of Microbiology, Prague, Czechoslovakia",
    },
    {
        "Prague Academy of Music",
    },
    {
        "University in Prague",
    },
    {
        "Film and TV School of Academy of Performing Arts, Prague",
    },
    {
        "Institute of Pharm & Biochem, Prague, Czech",
    },
    {
        "West Paulista University (UNOESTE)",
        "West Paulista University (UNOESTE), Brazil",
        "Universidade Estadual Paulista",
        "Universidade Estadual Paulista, Brazil",
    },
    {
        "Paulista University",
    },
    {
        "Von Karman Institute for Fluid Dynamics",
    },
    {
        "University of Sannio",
        "University of Sannio, Italy",
    },
    {
        "University of Salento, Lecce (Italy)",
    },
    {
        "University of Linz",
        "University of Linz, Austria",
    },
    {
        "University of Jaffna",
        "University of Jaffna, Sri Lanka",
    },
    {
        "University of Jodhpur", "University of Jodhpur, India",
    },
    {
        "ENSAE ParisTech",
    },
    {
        "ENST, Paris",
    },
    {
        "Mines Paris Tech",
    },
    {
        "Laboratory of Atomic and Molecular Collisions, Orsay, France",
    },
    {
        "Agro Paris Tech",
    },
    {
        "French National Centre for Scientific Research", "French National Centre for Scientific Research (CNRS)", "CNRS Gif-sur-Yvette, Paris, France",
    },
    {
        "Northwest Agricultural University, China", "Northwest Agricultural and Forestry University, China",
    },
    {
        "Rajendra Agricultural University", "Rajendra Agricultural University, India",
    },
    {
        "Tamil Nadu Agricultural University Coimbatore", "Tamil Nadu Agricultural University Coimbatore, India",
    },
    {
        "Academy Of Agriculture, Krakow, Poland", "Academy of Agriculture (Poland)",
    },
    {
        "Higher Institute of Medicine Sofia Medical University, Sofia", "Higher Medical University", "Higher Medical University, Bulgaria",
    },
    {
        "Basel School of Design", "Basel School of Design, Switzerland", "Schule fur Gestaltung, Basel, Switzerland",
    },
    {
        "Baroda",
    },
    {
        "Bass University",
    },
    {
        "Azabu University", "Azabu University, Japan",
    },
    {
        "Azabu Veterinary College",
    },
    {
        "Institute of Nuclear Physics of Lyon",
    },
    {
        "National Veterinary School of Lyon", "National Veterinary School of Lyon, France", "Ecole National Veterinaire de Lyon",
    },
    {
        "Universite Jean Moulin, Lyon III",
        "Jean Moulin University Lyon",
    },
    {
        "Faculte De Medicine Lyon, , France",
    },
    {
        "Universite Catholique de Lyon.",
    },
    {
        "Universite de Bretagne OccidentalUniversite de Bretagne Occidentale (Fra", "Universite de Bretagne OccidentalUniversite de Bretagne Occidentale Fra",
    },
    {
        "University of Illes Balears", "University of Illes Balears, Spain",
    },
    {
        "State Medical Institute",
    },
    {
        "State Medical Institute, Ukraine",
    },
    {
        "State Polytechnic University",
    },
    {
        "State Polytechnic University, Ukraine",
    },
    {
        "Strayer University",
    },
    {
        "Sukhadia University",
    },
    {
        "Brussels Conservatory", "Brussels Conservatory, Belgium",
    },
    {
        "Universities of Bonn, Brussels and Mannheim",
    },
    {
        "University Hassan II",
    },
    {
        "Al-Farabi Kazakh National University", "Al-Farabi Kazakh National University,  Kazakhstan",
    },
    {
        "Akdeniz Universitesi", "Akdeniz Universitesi, Turkey", "Akdeniz University, Turkey",
    },
    {
        "Akita University", "Akita University, Japan",
    },
    {
        "Aligarah University",
    },
    {
        "University of Sulaimani,Iraq",
    },
    {
        "Al-Anbar Medical College, Iraq",
    },
    {
        "Alaska Pacific University",
    },
    {
        "Catholic University of Lublin",
    },
    {
        "Madeira University",
    },
    {
        "Madonna University",
    },
    {
        "Madurai University",
    },
    {
        "Magadh University",
    },
    {
        "Magdalen College",
    },
    {
        "Mount Holyoke College", "MOUNT HOLYOKE COLLEGE",
    },
    {
        "Muhlenberg College",
    },
    {
        "Mumbai Veterinary College",
    },
    {
        "University of Rijeka",
        "University of Rijeka in Croatia",
    },
    {
        "University of Podova",
        "University of Podova, Italy",
    },
    {
        "University of Piemonte Orientale A. Avogadro",
        "University of Piemonte Orientale A. Avogadro, Italy",
    },
    {
        "Eastern Mediterranean University",
        "Eastern Mediterranean University, Turkey",
    },
    {
        "University of Mediterranean Medical College, Marseille, France",
        "University of Mediterranee Medical College In Marseille",
    },
    {
        "University of Meerut",
    },
    {
        "University of Merida Yucatan, Mexico",
    },
    {
        "University GHS Essen School of Medicine",
        "University GHS Essen School of Medicine, Germany",
    },
    {
        "University Hospital Centre of Zagreb",
    },
    {
        "University Carlos III de Madrid",
        "Health Institute Of Salud Carlos Iii, Madrid, Spain",
    },
    {
        "Universidad Veracruzana",
    },
    {
        "Universidad de San Martin de Porres",
        "Universidad de San Martin de Porres, Peru",
    },
    {
        "Universidad Nacional del Sur",
        "Universidad Nacional del Sur, Argentina",
    },
    {
        "Autonomous University of Puebla",
        "Universidad Popular Autonoma del Estado de Puebla",
    },
    {
        "B.A. Fort Lewis College",
    },
    {
        "Belarusian Polytechnic Institute",
    },
    {
        "Conservatory of Belarus",
    },
    {
        "Combs College of Music",
    },
    {
        "Colombian School of Odontology",
    },
    {
        "Cracow University",
        "Cracow University, Poland",
        "Krakow University",
    },
    {
        "Cumberland School of Law",
    },
    {
        "Dibrugarh University",
    },
    {
        "Dragomanov National Pedagogical University, Kiev",
    },
    {
        "Dropsie College",
        "Dropsie University",
    },
    {
        "Ecole du Louvre",
    },
    {
        "Erciyes University", "Erciyes University, Turkey",
    },
    {
        "Escuela Normal de ca Muguey",
        "Escuela Normal de ca Muguey, Cuba",
    },
    {
        "French National Institute for Agricultural Research",
        "French National Institute for Agricultural Research, France",
    },
    {
        "Frankfurt University",
        "Frankfurt University, Germany",
    },
    {
        "Federal University Rio Grande Norte, Natal",
    },
    {
        "Altai State University, Russia",
    },
    {
        "Alverno College",
    },
    {
        "Almaty State Medical University",
    },
    {
        "Argosy University San Francisco Bay Area",
    },
    {
        "Argosy University",
    },
    {
        "Argosy University, Phoenix",
    },
    {
        "Argosy University-Orange County",
    },
    {
        "American School of Professional Psychology at Argosy University",
    },
    {
        "Amirkabir University of Technology",
    },
    {
        "Diplomate American College of Veterinary Pathologists",
    },
    {
        "American College of Veterinary Anesthesiologists",
    },
    {
        "American College of Veterinary Clinical Pharmacology",
    },
    {
        "American College of Veterinary Internal Medicine",
    },
    {
        "American College of Veterinary Preventive Medicine",
    },
    {
        "American College of Veterinary Surgeons",
    },
    {
        "American College of Zoological Medicine",
    },
    {
        "American Board of Medical Genetics",
    },
    {
        "American Board of Radiology",
    },
    {
        "American Board of Urology",
    },
    {
        "Capital University of Economics and Business, Beijing",
    },
    {
        "Belfer Graduate School",
    },
    {
        "Birmingham School of Art in England",
    },
    {
        "Bogomolets National Medical University",
    },
    {
        "Bucaramanga, Colombia",
    },
    {
        "University of Caen, France",
        "Caen University",
        "Universite de CaenBasse-Normandie",
        "Universite de Mons",
    },
    {
        "the Institute of Biochemistry and Physiology"
    },
    {
        "the Institute of Genetics,"
    },
    {
        'Universitatea "Lucian Blaga" din Sibiu'
    },
    {
        'Institute of Mathematics "Simion Stoilow" of the Romanian Academy'
    },
    {
        "Romanian Academy, Cluj-Napoca, Romania"
    },
    {
        "Nizhny Novgorod State University, Russia",
        "Nizhny Novgorod Technical University",
        "Nizhny Novgorod Technical University, Russia",
        "University of Nizhny Novgorod",
        "University of Nizhny Novgorod, Russia"
    },
    {
        "University of Peradeniya",
        "University of Peradeniya, Sri Lanka"
    },
    {
        "University of Pau",
        "University of Pau, France"
    },
    {
        "University of Potenza",
        "University of Potenza, Italy"
    },
    {
        "University of Rabat",
        "University of Rabat, Morocco"
    },
    {
        "University of Sciences"
    },
    {
        "Pulkova Observatory, St. Petersburg"
    },
    {
        "Pushkin Institute",
        "Pushkin Russian Language Institute, Moscow"
    },
    {
        "State Hydrological Institute, Saint Petersburg"
        'Russian State Hydrometeorological University',
    },
    {
        "Saint Petersburg Pavlov Medical University",
        "St Petersburg State Pavlov Medical University"
        "Medical Academy, St. Petersburg, Russia"
    },
    {
        "St. Petersburg Academy of Sciences"
    },
    {
        "St. Petersburg Forest Academy"
    },
    {
        "St. Petersburg Institute of Technology"
    },
    {
        "St. Petersburg State Conservatory"
    },
    {
        "St. Petersburg State Pediatric Medical Academy",
        "St. Petersburg State Pediatric Medical Academy, Russia"
    },
    {
        "Tampere University of Technology",
        "Tampere University of Technology, Finland"
    },
    {
        "Technical University of Liberec",
        "Technical University of Liberec, Czech Republic"
    },
    {
        "Technical University Carolo - Wilhelmina, Braunschweig, Germany",
        "Technical University Carolo-Wilhelmina, Brunswick"
    },
    {
        "Technical University of Nova Scotia",
        "Technical University of Nova Scotia, Halifax"
    },
    {
        "Hubmoldt Universitat zu Berlin",
        "Hubmoldt Universitat zu Berlin, Germany"
    },
    {
        "University of the Arts Berlin", "University of the Arts Berlin, Germany"
    },
    {
        "Wroclaw Medical University, Poland"
    },
    {
        "Wroclaw University, Poland"
    },
    {
        "University of Thessaly",
        "University of Thessaly, Greece"
    },
    {
        "University of Fine Arts"
    },
    {
        "University of Dakar"
    },
    {
        "University of Cyprus"
    },
    {
        "University of Castilla La Mancha",
        "University of Castilla La Mancha, Spain"
    },
    {
        "University of Balamand",
        "University of Balamand, Lebanon"
    },
    {
        "University of Besancon",
        "University of Besancon, France"
    },
    {
        'Bulgarian Academy of Sciences',
        "University and Bulgarian Academy of Sciences",
        "University and Bulgarian Academy of Sciences, Bulgaria"
    },
    {
        "UniversitT Stendhal Grenoble III",
        "Université Stendhal Grenoble III, France"
    },
    {
        "Universidad Nacional de San Luis",
        "Universidad Nacional de San Luis, Argentina"
    },
    {
        "National University of Litoral",
        "University Nacional del Litoral"
    },
    {
        "Universidad Nacional De Buenos Aires (Argentina)",
        "National University of Buenos Aires"
    },
    {
        "Universidad Autónoma de Chihuahua"
    },
    {
        "Universidad Autonoma Del Edo De Mexico, Toluca"
    },
    {
        "Universidad Antonio de Nebrija",
        "Universidad Antonio de Nebrija, Spain"
    },
    {
        "USSR - Free-Standing Inst"
    },
    {
        "USSR - Non-Medical School"
    },
    {
        "International Christian University, Tokyo"
    },
    {
        "Tokyo College of Pharmacy", "Tokyo College of Pharmacy, Japan"
    },
    {
        "Tokyo Inst of Tech-Tokyo"
    },
    {
        "Tokyo National University of Fine Arts and Music"
    },
    {
        "Tokyo University of Marine Science and Technology "
    },
    {
        "Thessaloniki Medical School",
        "Thessaloniki Medical School, Greece"
    },
    {
        "Terna Medical College",
        "Terna Medical College, India"
    },
    {
        "Technical University"
    },
    {
        "State Technical University, Moscow"
    },
    {
        "State Academy of Theatrical Arts"
    },
    {
        "Zoological Institute, Academy of Science, St. Petersburg, Russia"
    },
    {
        "Split University School of Medicine",
        "Split University School of Medicine, Croatia"
    },
    {
        "Southern Cross University", "Southern Cross University, Australia"
    },
    {
        "Siena University School of Medicine",
        "Siena University School of Medicine, Italy"
    },
    {
        "Seikei University",
        "Seikei University, Japan"
    },
    {
        "School of Sciences Montevideo",
        "School of Sciences Montevideo, Uruguay"
    },
    {
        "SN Bose National Centre for Basic Sciences",
        "SN Bose National Centre for Basic Sciences, India"
    },
    {
        "Netaji Subhash Chandra Bose Medical College",
        "Netaji Subhash Chandra Bose Medical College, India"
    },
    {
        "National Research Institute"
    },
    {
        "National Research University"
    },
    {
        "National Defense Medical College",
        "National Defense Medical College, Japan"
    },
    {
        "National Gallery of Victoria Art School",
        "National Gallery of Victoria Art School, Australia"
    },
    {
        "National College of Art and Design", "National College of Art and Design, Ireland"
    },
    {
        "Maharaja Krishna Chandra Gajapati Medical College, Orissa, India"
    },
    {
        "Lviv National University",
        "Lviv Polytechnic Institute",
        "Lviv Polytechnic National University"
    },
    {
        "Lille Medical School",
        "Lille Medical School, France"
    },
    {
        "Lappeenranta University of Technology",
        "Lappeenranta University of Technology, Finland"
    },
    {
        "Kiev Research Institute of Urology and Nephrology",
        "Kiev Research Institute of Urology and Nephrology, Ukraine"
    },
    {
        "International University"
    },
    {
        "Institute of Virology"
    },
    {
        "Indian Statistical Institute, Calcutta"
    },
    {
        "Indian Statistical Institute, Delhi"
    },
    {
        "Indian Statistical Institute, India"
    },
    {
        "Ian Amos Comenius University",
        "Ian Amos Comenius University, Czechoslovakia"
    },
    {
        "Henley Management Colleg",
        "Henley Management College, England"
    },
    {
        "Helmholtz Center Munich"
    },
    {
        "Helmholtz Centre Potsdam GFZ German Research Centre for Geosciences"
    },
    {
        "Harokopio University", "Harokopio University, Greece"
    },
    {
        "Japan Graduate University for Advanced Studies",
        "Graduate University of Advanced Studies in Japan",
        "Japan Graduate University"
    },
    {
        "Gr. T. Popa University Of Iasi",
        "Gr. T. Popa University Of Iasi, Romania"
    },
    {
        "Goa University, Goa Medical College, India",
        "Goa University, India"
    },
    {
        "Glushkov Cybernetics Institute",
        "Glushkov Institute Of Cybernetics, Kiev, Ukraine",
        "Institute for Cybernetics, Ukrainian Academy of Science"
    },
    {
        "College of Europe"
    },
    {
        "College of France"
    },
    {
        "College of Medicine"
    },
    {
        "Colegio de México",
        "College Of Mexico"
    },
    {
        "Chang Gung University",
        "Chang Gung University, Taiwan"
    },
    {
        "Centre Hospitalier Universitaire Vaudois, Lausanne",
        "Lausanne Medical School, Switzerland"
    },
    {
        "Central Aerological Observatory, Moscow",
        "Central Aerological Observatory, Moscow, Russia"
    },
    {
        "Center for Advanced Biotechnology and Medicine"
    },
    {
        "California College of Podiatric Medicine"
    },
    {
        "California Graduate School of Theology"
    },
    {
        "California School of Professional Psychology, Alameda"
    },
    {
        "California School of Professional Psychology, Berkeley"
    },
    {
        "Bolan Medical College, Quetta Pakistan"
    },
    {
        "Autonomous University of Tamaulipas"
    },
    {
        "Autonomous National University of Honduras"
    },
    {
        "Austrailian National University", "Australia National University"
    },
    {
        "Australia - Non-Medical School"
    },
    {
        "Arkhangelsk Medical Institute, Arkhangelsk"
    },
    {
        "Anahuac University School of Medicine",
        "Anahuac University School of Medicine, Mexico"
    },
    {
        "American Association of University"
    },
]

ALL_LOW_CONFIDENCE_ALIASES = SET_MATCHED_DEGREE_ALIASES + DUPLICATES_ALIASES + MISCELLANEOUS_ALIASES
