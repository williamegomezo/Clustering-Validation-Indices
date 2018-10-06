import './layout.scss'
import Layout from './../../data/Layout.json'
import MainMenu from './../modules/mainMenu/main-menu'

// Components of the page
const main = document.querySelector('.leftNavbar')

/* eslint-disable no-new */
const menuNode = document.createElement('DIV')
main.appendChild(menuNode)
new MainMenu(menuNode, Layout.navbar.components.menu)
