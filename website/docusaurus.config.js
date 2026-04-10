// @ts-check
const { themes } = require('prism-react-renderer');

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Melody Match',
  tagline: 'Classical Music Similarity Search using MERT + LoRA',
  favicon: 'img/favicon.ico',

  url: 'https://mperumal-usd.github.io',
  baseUrl: '/capstone_team_3/',

  organizationName: 'mperumal-usd',
  projectName: 'capstone_team_3',

  onBrokenLinks: 'warn',
  onBrokenMarkdownLinks: 'warn',

  markdown: {
    mermaid: true,
  },

  themes: ['@docusaurus/theme-mermaid'],

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          routeBasePath: '/',
        },
        blog: false,
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      image: 'img/social-card.png',
      colorMode: {
        defaultMode: 'dark',
        disableSwitch: false,
        respectPrefersColorScheme: true,
      },
      mermaid: {
        theme: { light: 'neutral', dark: 'dark' },
      },
      navbar: {
        title: 'Melody Match',
        logo: {
          alt: 'University of San Diego',
          src: 'img/logo-usd.png',
          style: { filter: 'brightness(0) invert(1)' },   // white on dark navbar
        },
        items: [
          { to: '/', label: 'Overview', position: 'left' },
          { to: '/pipeline', label: 'Pipeline', position: 'left' },
          { to: '/models', label: 'Models', position: 'left' },
          { to: '/results', label: 'Results', position: 'left' },
          { to: '/tools', label: 'Tools', position: 'left' },
          { to: '/notebooks', label: 'Notebooks', position: 'left' },
          { to: '/team', label: 'Team', position: 'left' },
          {
            href: 'https://github.com/mperumal-usd/capstone_team_3',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Docs',
            items: [
              { label: 'Overview', to: '/' },
              { label: 'Pipeline', to: '/pipeline' },
              { label: 'Models', to: '/models' },
              { label: 'Results', to: '/results' },
              { label: 'Tools', to: '/tools' },
            ],
          },
          {
            title: 'Resources',
            items: [
              { label: 'GitHub', href: 'https://github.com/mperumal-usd/capstone_team_3' },
              { label: 'MERT Paper', href: 'https://arxiv.org/abs/2306.00107' },
              { label: 'Open in Colab', href: 'https://colab.research.google.com/github/mperumal-usd/capstone_team_3/blob/main/notebooks/COLAB_MERT_Finetune_v5.ipynb' },
            ],
          },
          {
            title: 'University of San Diego',
            items: [
              { label: 'AAI-590 Capstone', href: 'https://www.sandiego.edu' },
              { label: 'Applied AI Program', href: 'https://www.sandiego.edu' },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Capstone Team 3 — University of San Diego, AAI-590.`,
      },
      prism: {
        theme: themes.github,
        darkTheme: themes.dracula,
        additionalLanguages: ['python', 'bash'],
      },
    }),
};

module.exports = config;
